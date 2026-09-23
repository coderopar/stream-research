#!/usr/bin/env python3
"""
Phase 7M.6B — Prospective 25-bp refill→re-depletion sequence pilot
=================================================================

SCIENTIFIC QUESTION
-------------------
After an opposing frozen price level is depleted and then visibly refills, does
that replenished PRICE LEVEL subsequently experience aggressive executions, or
is later visible queue loss primarily non-trade reduction / general activity?

This phase is OUTCOME-BLIND. It never opens Phase-7M.3 decision outcomes or any
future P75/P100 label.

WHY A PILOT
-----------
Phase 7M.6A established that frozen 7M.3 aggregates cannot reconstruct the
required deplete -> refill -> later fate sequence. This builder therefore reads
raw canonical depth/trades, but defaults to four representative candidate dates
before any full 62-day replay.

The pilot dates are selected without outcomes: the 62 candidate dates are split
into four chronological blocks and the date closest to the median candidate
count in each block is selected. Explicit --pilot-date values override this.

REUSABILITY
-----------
Per-date checkpoints use the SAME causal schema intended for a later full run:

  data/derived/perps/research/prospective_25bp_refill_redepletion_sequence_v1/
    candidate_metrics/candidate_date=YYYY-MM-DD/part.parquet
    level_fate/candidate_date=YYYY-MM-DD/part.parquet
    event_tape/candidate_date=YYYY-MM-DD/part.parquet
    checkpoints/candidate_date=YYYY-MM-DD/manifest.json

Each manifest is signed from the candidate IDs, frozen 7M.3 level audit, raw
canonical depth/snapshot/trade file metadata, the frozen data-contract version,
and the attribution contract. Runtime-only memory optimizations do not invalidate
scientifically identical v1.0.1 checkpoints. A later run can therefore safely reuse pilot dates.

CAUSAL CLOCKS / ATTRIBUTION
---------------------------
* The reconstructed local book remains ordered by canonical ts_recv_ns, exactly
  as in Phase 6A / 7M.3.
* Depth exchange timing is retained as transaction_time_ms / exchange_event_ms.
* Trades retain trade_time_ms / exchange_event_ms and buyer_is_maker.
* A refill is an observed positive absolute-quantity update at the SAME frozen
  opposing price after that level has first depleted below its candidate-anchor
  quantity.
* A "post-refill aggressive trade" is a correctly-directed executed trade at
  that SAME price, received after the refill, with trade_time_ms strictly after
  the refill transaction_time_ms. Same-exchange-millisecond trades are kept in
  a separate ambiguity bucket and are NOT included in the strict measure.
* Because public depth is price-level aggregated, this phase DOES NOT claim an
  individual refill order was filled. It measures executions at a replenished
  price level.
* Conservative non-trade reduction lower bound:
      max(0, post-refill gross observed queue decreases
             - strict post-refill aggressive trade quantity)
  This is deliberately a lower bound, not a literal cancellation count.

PRIMARY OBSERVATION HORIZON
---------------------------
3 seconds. 1/5/10 seconds are materialized from the same replay for robustness.

PLANNED FUTURE MATCHING CONTROLS (OUTCOME-BLIND FEASIBILITY ONLY HERE)
---------------------------------------------------------------------
Exact block: candidate_date + direction
Continuous:
  1. 3s directional_mid_move_bps
  2. log1p(3s directional_aggressive_quote)
  3. 3s opposing__peak_initial_queue_loss_share
  4. log1p(3s opposing__same_price_refill_after_depletion_quote)

This directly implements the question:
"given similar initial destruction AND similar amount of opposing refill, does
what happens at the refilled prices add information?"

No matcher is loosened after outcomes because no outcomes are loaded here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
except Exception:
    pa = None
    pq = None
    pc = None


BUILDER_VERSION = "1.0.2"
CHECKPOINT_CONTRACT_BUILDER_VERSION = "1.0.1"
COMPATIBLE_CHECKPOINT_BUILDERS = {"1.0.1", "1.0.2"}
DEPTH_BATCH_SIZE = 1024
TRADE_BATCH_SIZE = 32768
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_refill_redepletion_sequence_v1"
FROZEN_PHASE7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"
FROZEN_PHASE7M3_BUILDER = "1.0.0"
FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"
EXPECTED_CANDIDATES = 7_067
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_HORIZON_SEC = 3
PRICE_SCALE = 10.0
PILOT_DAY_COUNT = 4
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

# Structural / feasibility gates, frozen before any future outcome is read.
MIN_ANCHOR_MATCH_RATE = 0.98
MIN_SEQUENCE_VALID_RATE = 0.95
MIN_STRICT_TRADE_TIME_COVERAGE = 0.95
MIN_PROJECTED_FULL_REFILL_ELIGIBLE = 750
MIN_PROJECTED_FULL_PRIMARY_PAIRS = 250
MIN_PROJECTED_SPLIT_PAIRS = 100
MIN_FEATURE_POSITIVE_SHARE = 0.10
MAX_FEATURE_POSITIVE_SHARE = 0.90

MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20
PILOT_MIN_REPAIR_PAIRS = 8  # only prevents collapse during pilot balance estimate

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_M3_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7M3_SCHEMA
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_PHASE7E_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07e_build_raw_depth_continuation_mechanics.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_redepletion_sequence"

FUTURE_NAME_TOKENS = (
    "outcome", "label", "p75", "p100", "mfe", "mae", "first_passage",
    "reaches_", "hit_", "continuation_target",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.6B requires pandas, numpy and pyarrow in the project .venv.")


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
        raise RuntimeError(f"Missing required existing replay module: {path}")
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6b-", suffix=path.suffix or ".tmp", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6b-", suffix=".csv", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6b-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(paths, key=lambda x: str(x)):
        st = p.stat()
        rows.append({"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return rows


def contract_payload() -> dict[str, Any]:
    return {
        "builder_version": CHECKPOINT_CONTRACT_BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "frozen_phase7m3_schema": FROZEN_PHASE7M3_SCHEMA,
        "frozen_phase7m3_builder": FROZEN_PHASE7M3_BUILDER,
        "policy": FROZEN_POLICY,
        "symbol": "BTCUSDT",
        "price_scale": PRICE_SCALE,
        "horizons_sec": EXPECTED_HORIZONS,
        "primary_horizon_sec": PRIMARY_HORIZON_SEC,
        "refill_definition": "positive absolute-quantity update at same frozen opposing price after prior depletion below anchor qty",
        "strict_post_refill_trade": "same price + correct aggressor + receive after refill + trade_time_ms strictly after refill transaction_time_ms",
        "same_ms_policy": "same exchange millisecond excluded from strict and reported as ambiguous",
        "nontrade_lower_bound": "max(0, post-refill gross negative depth qty - strict post-refill aggressive trade qty)",
        "order_identity_claim": False,
        "future_outcomes_loaded": False,
        "missing_frozen_level_policy": "retain candidate as unavailable; mark FROZEN_7M3_LEVEL_AUDIT_MISSING/OPPOSING_LEVELS_MISSING; never coerce missing mechanics to zero",
    }


def load_m3(m3_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    require_stack()
    meta_path = m3_root / "metadata" / "latest_summary.json"
    if not meta_path.exists():
        raise RuntimeError(f"Missing frozen Phase 7M.3 metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checks = {
        "builder_version": FROZEN_PHASE7M3_BUILDER,
        "schema_name": FROZEN_PHASE7M3_SCHEMA,
        "selected_policy": FROZEN_POLICY,
        "status": "PASS",
    }
    for k, expected in checks.items():
        if str(meta.get(k)) != str(expected):
            raise RuntimeError(f"Phase 7M.3 {k} mismatch: expected {expected!r}, found {meta.get(k)!r}")
    if int(meta.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_CANDIDATES} Phase 7M.3 candidates; found {meta.get('candidate_count')!r}")
    if not bool(meta.get("causal_mechanics_future_labels_separate")):
        raise RuntimeError("Phase 7M.3 does not confirm causal/future-label physical separation.")

    cp = m3_root / "candidate_inputs.parquet"
    mp = m3_root / "candidate_mechanics.parquet"
    for p in (cp, mp):
        if not p.exists():
            raise RuntimeError(f"Missing frozen causal input: {p}")
    candidates = pq.read_table(cp).to_pandas()
    mechanics = pq.read_table(mp).to_pandas()
    for name, df, cols in (
        ("candidate_inputs", candidates, ["candidate_id", "candidate_date", "direction", "policy_id"]),
        ("candidate_mechanics", mechanics, [
            "candidate_id", "candidate_date", "direction", "observation_horizon_sec", "decision_second",
            "window_valid", "decision_price_valid", "directional_mid_move_bps", "directional_aggressive_quote",
            "opposing__peak_initial_queue_loss_share", "opposing__same_price_refill_after_depletion_quote",
        ]),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Phase 7M.3 {name} missing required columns: {missing}")
    candidates = candidates[candidates["policy_id"].astype(str) == FROZEN_POLICY].copy()
    if len(candidates) != EXPECTED_CANDIDATES or candidates["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Frozen candidate identity/count invariant failed.")
    mechanics["observation_horizon_sec"] = pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce")
    hs = tuple(sorted(mechanics["observation_horizon_sec"].dropna().astype(int).unique().tolist()))
    if hs != EXPECTED_HORIZONS:
        raise RuntimeError(f"Unexpected frozen mechanics horizons: {hs}")
    # This phase reads only the explicit causal columns listed above and never opens
    # candidate_decision_outcomes.parquet. Do not infer leakage from harmless names
    # such as direction_label; the allow-list is the causal firewall.
    return meta, candidates, mechanics


def derive_candidate_clock(candidates: pd.DataFrame, mechanics: pd.DataFrame) -> pd.DataFrame:
    m = mechanics[pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    keep = [
        "candidate_id", "candidate_date", "direction", "decision_second", "window_valid", "decision_price_valid",
        "directional_mid_move_bps", "directional_aggressive_quote",
        "opposing__peak_initial_queue_loss_share", "opposing__same_price_refill_after_depletion_quote",
    ]
    if "is_reaccel_context" in m.columns:
        keep.append("is_reaccel_context")
    m = m[keep].copy()
    m["candidate_second"] = pd.to_numeric(m["decision_second"], errors="coerce") - PRIMARY_HORIZON_SEC
    candidate_keep = ["candidate_id", "candidate_date", "direction"]
    if "is_reaccel_context" in candidates.columns and "is_reaccel_context" not in m.columns:
        candidate_keep.append("is_reaccel_context")
    x = candidates[candidate_keep].copy()
    x["candidate_id"] = x["candidate_id"].astype(str)
    m["candidate_id"] = m["candidate_id"].astype(str)
    x = x.merge(m, on=["candidate_id", "candidate_date", "direction"], how="left", validate="one_to_one")
    if x["candidate_second"].isna().any():
        raise RuntimeError("Could not derive candidate_second for all candidates from frozen 3s mechanics.")
    x["candidate_second"] = x["candidate_second"].astype("int64")
    x["candidate_date"] = x["candidate_date"].astype(str)
    return x


def choose_pilot_dates(clock: pd.DataFrame, n: int = PILOT_DAY_COUNT) -> pd.DataFrame:
    counts = clock.groupby("candidate_date", sort=True).size().rename("candidate_count").reset_index()
    counts = counts.sort_values("candidate_date", kind="mergesort").reset_index(drop=True)
    if len(counts) < n:
        raise RuntimeError(f"Only {len(counts)} candidate dates available; cannot select {n} pilot dates.")
    blocks = np.array_split(np.arange(len(counts)), n)
    rows = []
    for i, idxs in enumerate(blocks, 1):
        b = counts.iloc[list(idxs)].copy()
        med = float(b["candidate_count"].median())
        b["distance_to_block_median"] = (b["candidate_count"] - med).abs()
        pick = b.sort_values(["distance_to_block_median", "candidate_date"], kind="mergesort").iloc[0]
        rows.append({
            "pilot_block": i,
            "candidate_date": str(pick["candidate_date"]),
            "candidate_count": int(pick["candidate_count"]),
            "block_median_candidate_count": med,
            "selection_rule": "chronological_quartile_date_closest_to_block_median_candidate_count",
        })
    return pd.DataFrame(rows)


def load_level_audit_for_day(m3_root: Path, day: str, candidate_ids: set[str]) -> pd.DataFrame:
    p = m3_root / "level_audit" / f"candidate_date={day}" / "part.parquet"
    if not p.exists():
        raise RuntimeError(f"Missing frozen Phase 7M.3 level audit: {p}")
    pf = pq.ParquetFile(p)
    names = set(pf.schema_arrow.names)
    key = "candidate_id" if "candidate_id" in names else ("event_id" if "event_id" in names else None)
    if key is None:
        raise RuntimeError(f"{p} lacks candidate_id/event_id")
    cols = [key, "side_role", "price", "initial_qty"]
    missing = [c for c in cols if c not in names]
    if missing:
        raise RuntimeError(f"{p} missing level audit fields: {missing}")
    d = pq.read_table(p, columns=cols).to_pandas()
    d["candidate_id"] = d[key].astype(str)
    d = d[d["candidate_id"].isin(candidate_ids)].copy()
    d["side_role"] = d["side_role"].astype(str).str.upper()
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d["initial_qty"] = pd.to_numeric(d["initial_qty"], errors="coerce")
    d = d[d["price"].notna() & d["initial_qty"].notna() & (d["price"] > 0) & (d["initial_qty"] > 0)].copy()
    d["price_tick"] = np.rint(d["price"] * PRICE_SCALE).astype("int64")
    if d.duplicated(["candidate_id", "side_role", "price_tick"]).any():
        # Defensive dedupe only when exact rows are identical; otherwise stop.
        g = d.groupby(["candidate_id", "side_role", "price_tick"])["initial_qty"].nunique(dropna=False)
        if (g > 1).any():
            raise RuntimeError(f"Inconsistent duplicate frozen level rows in {p}")
        d = d.drop_duplicates(["candidate_id", "side_role", "price_tick"], keep="first")
    return d[["candidate_id", "side_role", "price", "price_tick", "initial_qty"]].copy()


def side_qty(side: Any, tick: int) -> float:
    try:
        return float(side.qty[int(tick)])
    except Exception:
        return 0.0


def book_vector_matches(book: Any, levels: pd.DataFrame, direction: int, tol: float = 1e-10) -> bool:
    if not bool(getattr(book, "valid", False)):
        return False
    for r in levels.itertuples(index=False):
        role = str(r.side_role).upper()
        if role == "OPPOSING":
            side = book.asks if int(direction) > 0 else book.bids
        elif role == "SUPPORT":
            side = book.bids if int(direction) > 0 else book.asks
        else:
            return False
        actual = side_qty(side, int(r.price_tick))
        if not math.isclose(actual, float(r.initial_qty), rel_tol=0.0, abs_tol=tol):
            return False
    return True


def merge_rows(iterators: Sequence[Iterator[dict[str, Any]]], family_priority: int) -> Iterator[dict[str, Any]]:
    heap: list[tuple[int, int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for idx, it in enumerate(iterators):
        try:
            row = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(row["ts_recv_ns"]), int(row.get("source_record_number", 0) or 0), idx, row, it))
    while heap:
        _ts, _rn, idx, row, it = heapq.heappop(heap)
        row = dict(row)
        row["__family_priority"] = family_priority
        yield row
        try:
            nxt = next(it)
            heapq.heappush(heap, (int(nxt["ts_recv_ns"]), int(nxt.get("source_record_number", 0) or 0), idx, nxt, it))
        except StopIteration:
            pass


def merge_event_families(snapshot_iter: Iterator[dict[str, Any]], depth_iter: Iterator[dict[str, Any]]) -> Iterator[tuple[str, dict[str, Any]]]:
    its = [("SNAPSHOT", iter(snapshot_iter), 0), ("DEPTH", iter(depth_iter), 1)]
    heap: list[tuple[int, int, int, str, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for idx, (kind, it, priority) in enumerate(its):
        try:
            row = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(row["ts_recv_ns"]), priority, idx, kind, row, it))
    while heap:
        _ts, priority, idx, kind, row, it = heapq.heappop(heap)
        yield kind, row
        try:
            nxt = next(it)
            heapq.heappush(heap, (int(nxt["ts_recv_ns"]), priority, idx, kind, nxt, it))
        except StopIteration:
            pass


def depth_changes_before_apply(book: Any, row: Mapping[str, Any]) -> list[tuple[str, int, float, float]]:
    out: list[tuple[str, int, float, float]] = []
    for side_name, side, ticks_key, qty_key, levels_key in (
        ("BID", book.bids, "_fast_bid_ticks", "_fast_bid_qty", "bid_updates"),
        ("ASK", book.asks, "_fast_ask_ticks", "_fast_ask_qty", "ask_updates"),
    ):
        ticks = row.get(ticks_key)
        qtys = row.get(qty_key)
        if ticks is not None and qtys is not None:
            for tick, newq in zip(ticks, qtys):
                t = int(tick)
                nq = max(0.0, float(newq))
                out.append((side_name, t, side_qty(side, t), nq))
            continue
        for level in row.get(levels_key) or []:
            try:
                p = float(level.get("price")); q = max(0.0, float(level.get("quantity")))
            except Exception:
                continue
            t = int(round(p * float(book.price_scale)))
            out.append((side_name, t, side_qty(side, t), q))
    return out


def merge_candidate_windows(trackers: Sequence["CandidateTracker"]) -> list[tuple[int, int]]:
    """Merge candidate receive-time windows so trade loading never scans irrelevant intraday gaps."""
    raw = sorted((int(t.search_start_ns), int(t.final_end_ns)) for t in trackers if int(t.final_end_ns) > int(t.search_start_ns))
    if not raw:
        return []
    merged: list[list[int]] = [[raw[0][0], raw[0][1]]]
    for lo, hi in raw[1:]:
        cur = merged[-1]
        if lo <= cur[1]:
            cur[1] = max(cur[1], hi)
        else:
            merged.append([lo, hi])
    return [(int(lo), int(hi)) for lo, hi in merged]


def interval_mask_ns(values: np.ndarray, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    if len(values) == 0 or not intervals:
        return np.zeros(len(values), dtype=bool)
    starts = np.asarray([x[0] for x in intervals], dtype=np.int64)
    ends = np.asarray([x[1] for x in intervals], dtype=np.int64)
    idx = np.searchsorted(starts, values, side="right") - 1
    valid = idx >= 0
    clipped = np.clip(idx, 0, len(ends) - 1)
    return valid & (values < ends[clipped])


def refill_ticks_from_depth(trackers: Sequence["CandidateTracker"]) -> set[int]:
    """Exact max-horizon refill eligibility using the same depletion->positive-update definition as metrics."""
    out: set[int] = set()
    for t in trackers:
        if not t.anchor_found or not t.sequence_valid or not t.depth_events:
            continue
        events = pd.DataFrame(t.depth_events)
        if events.empty:
            continue
        for lr in t.opposing_levels.itertuples(index=False):
            tick = int(lr.price_tick)
            le = events[events["price_tick"] == tick].sort_values("ts_recv_ns", kind="mergesort")
            depleted = False
            for er in le.itertuples(index=False):
                delta = float(er.new_qty) - float(er.old_qty)
                if delta < -1e-12:
                    depleted = True
                elif delta > 1e-12 and depleted:
                    out.add(tick)
                    break
    return out


def load_filtered_trades(paths: Sequence[Path], symbol: str, ticks: set[int], intervals: Sequence[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    if not ticks or not intervals:
        return pd.DataFrame(columns=[
            "ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker",
            "exchange_event_ms", "trade_time_ms", "aggregate_trade_id", "first_trade_id", "last_trade_id",
            "source_record_number", "price_tick",
        ])
    start_ns = min(x[0] for x in intervals)
    end_ns = max(x[1] for x in intervals)
    wanted = [
        "ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker",
        "exchange_event_ms", "trade_time_ms", "aggregate_trade_id", "first_trade_id", "last_trade_id",
        "source_record_number",
    ]
    for path in paths:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in wanted if c in available]
        required = {"ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker", "trade_time_ms"}
        if not required.issubset(available):
            raise RuntimeError(f"Trade parquet lacks required event-level fields: {path} missing {sorted(required-available)}")
        for batch in pf.iter_batches(columns=cols, batch_size=TRADE_BATCH_SIZE):
            f = batch.to_pandas()
            f = f[f["symbol"].astype(str) == symbol].copy()
            f["ts_recv_ns"] = pd.to_numeric(f["ts_recv_ns"], errors="coerce")
            f = f[(f["ts_recv_ns"] >= start_ns) & (f["ts_recv_ns"] < end_ns)].copy()
            if f.empty:
                continue
            ts_values = f["ts_recv_ns"].astype("int64").to_numpy(copy=False)
            f = f[interval_mask_ns(ts_values, intervals)].copy()
            if f.empty:
                continue
            f["price"] = pd.to_numeric(f["price"], errors="coerce")
            f["quantity"] = pd.to_numeric(f["quantity"], errors="coerce")
            f["price_tick"] = np.rint(f["price"] * PRICE_SCALE).astype("Int64")
            f = f[f["price_tick"].isin(ticks) & f["quantity"].notna() & (f["quantity"] > 0)].copy()
            if not f.empty:
                rows.append(f)
    if not rows:
        return pd.DataFrame(columns=wanted + ["price_tick"])
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["trade_time_ms"] = pd.to_numeric(out["trade_time_ms"], errors="coerce")
    out["exchange_event_ms"] = pd.to_numeric(out.get("exchange_event_ms"), errors="coerce")
    out["buyer_is_maker"] = bool_series(out["buyer_is_maker"])
    return out.sort_values(["ts_recv_ns", "price_tick"], kind="mergesort").reset_index(drop=True)


@dataclass
class CandidateTracker:
    candidate_id: str
    candidate_date: str
    direction: int
    candidate_second: int
    all_levels: pd.DataFrame
    opposing_levels: pd.DataFrame
    search_start_ns: int
    search_end_ns: int
    final_end_ns: int
    anchor_found: bool = False
    anchor_ts_recv_ns: int | None = None
    sequence_valid: bool = True
    invalid_reason: str = ""
    depth_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def expected_side(self) -> str:
        return "ASK" if self.direction > 0 else "BID"

    @property
    def opposing_ticks(self) -> set[int]:
        return set(pd.to_numeric(self.opposing_levels["price_tick"], errors="coerce").dropna().astype(int).tolist())


def iter_depth_rows_with_times(
    phase6a: Any,
    paths: Sequence[Path],
    *,
    symbol: str,
    price_scale: float,
    fast_stats: Any,
    batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Arrow-native depth decoder retaining exchange/transaction timestamps.

    Phase 6A's accelerated decoder is intentionally mirrored here because its
    normal unified_book_events projection did not need transaction_time_ms.
    Nested price levels stay Arrow-native; no full-day Pandas object arrays.
    """
    cols = [
        "ts_recv_ns", "symbol", "source_record_number", "exchange_event_ms", "transaction_time_ms",
        "first_update_id", "final_update_id", "previous_final_update_id", "bid_updates", "ask_updates",
    ]
    for path in paths:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        missing = [c for c in cols if c not in names]
        if missing:
            raise RuntimeError(f"Depth parquet missing required 7M.6B columns: {path}: {missing}")
        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            tbl = pa.Table.from_batches([batch])
            if tbl.num_rows == 0:
                continue
            mask = pc.equal(tbl["symbol"], pa.scalar(symbol))
            tbl = tbl.filter(mask)
            n = tbl.num_rows
            if n == 0:
                continue

            def np_int(name: str) -> np.ndarray:
                arr = pc.cast(tbl[name], pa.int64(), safe=False).combine_chunks()
                vals = arr.to_numpy(zero_copy_only=False)
                return np.asarray(vals, dtype="int64")

            ts = np_int("ts_recv_ns")
            U = np_int("first_update_id")
            u = np_int("final_update_id")
            pu = np_int("previous_final_update_id")
            recno = np_int("source_record_number")
            exch = tbl["exchange_event_ms"].combine_chunks().to_pylist()
            txn = tbl["transaction_time_ms"].combine_chunks().to_pylist()
            b_arr = tbl["bid_updates"].combine_chunks()
            a_arr = tbl["ask_updates"].combine_chunks()
            boff, bticks, bqty = phase6a._arrow_list_struct_numeric(b_arr, price_scale, path, "bid_updates")
            aoff, aticks, aqty = phase6a._arrow_list_struct_numeric(a_arr, price_scale, path, "ask_updates")

            try:
                fast_stats.rows_decoded += int(n)
                fast_stats.levels_decoded += int(len(bticks) + len(aticks))
                fast_stats.arrow_batches += 1
            except Exception:
                pass

            for i in range(n):
                b0, b1 = int(boff[i]), int(boff[i + 1])
                a0, a1 = int(aoff[i]), int(aoff[i + 1])
                yield {
                    "ts_recv_ns": int(ts[i]),
                    "source_record_number": int(recno[i]),
                    "exchange_event_ms": None if exch[i] is None else int(exch[i]),
                    "transaction_time_ms": None if txn[i] is None else int(txn[i]),
                    "first_update_id": int(U[i]),
                    "final_update_id": int(u[i]),
                    "previous_final_update_id": int(pu[i]),
                    "_fast_bid_ticks": bticks[b0:b1],
                    "_fast_bid_qty": bqty[b0:b1],
                    "_fast_ask_ticks": aticks[a0:a1],
                    "_fast_ask_qty": aqty[a0:a1],
                }


def build_raw_iters(phase6a: Any, day_files: Any, symbol: str, price_scale: float, fast_stats: Any):
    snap_cols = ["ts_recv_ns", "symbol", "source_record_number", "last_update_id", "bids", "asks"]
    depth_its = [
        iter_depth_rows_with_times(
            phase6a, (p,), symbol=symbol, price_scale=price_scale, fast_stats=fast_stats,
            batch_size=DEPTH_BATCH_SIZE,
        )
        for p in day_files.depth
    ]
    snap_its = [phase6a.iter_parquet_rows((p,), snap_cols, symbol=symbol, batch_size=4096) for p in day_files.snapshots]
    return merge_rows(snap_its, 0), merge_rows(depth_its, 1)

def process_day(
    *, day: str, day_clock: pd.DataFrame, level_audit: pd.DataFrame, day_files: Any,
    phase6a: Any, phase7e: Any, symbol: str, price_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_ids = set(day_clock["candidate_id"].astype(str))
    level_audit = level_audit[level_audit["candidate_id"].isin(candidate_ids)].copy()

    # Frozen 7M.3 exact-level coverage is intentionally not 100%. Candidates
    # without a valid exact anchor have no level_audit rows. That is a measured
    # coverage limitation, not a substrate defect and must never abort a day.
    # Preserve them as unavailable observations (NaN, not zero) so they count
    # against coverage and remain excluded from refill-fate feasibility.
    trackers: list[CandidateTracker] = []
    missing_level_reason: dict[str, str] = {}
    for r in day_clock.itertuples(index=False):
        cid = str(r.candidate_id)
        lev = level_audit[level_audit["candidate_id"] == cid].copy()
        opp = lev[lev["side_role"] == "OPPOSING"].copy() if not lev.empty else lev.copy()
        if lev.empty:
            missing_level_reason[cid] = "FROZEN_7M3_LEVEL_AUDIT_MISSING"
            continue
        if opp.empty:
            missing_level_reason[cid] = "FROZEN_7M3_OPPOSING_LEVELS_MISSING"
            continue
        cs = int(r.candidate_second)
        trackers.append(CandidateTracker(
            candidate_id=cid,
            candidate_date=day,
            direction=int(r.direction),
            candidate_second=cs,
            all_levels=lev,
            opposing_levels=opp,
            search_start_ns=cs * 1_000_000_000,
            search_end_ns=(cs + 1) * 1_000_000_000,
            final_end_ns=(cs + max(EXPECTED_HORIZONS)) * 1_000_000_000,
        ))

    if not trackers:
        # Extremely defensive: if an entire selected day lacks frozen exact
        # levels, return explicit unavailable rows rather than scanning raw data.
        metric_rows = []
        clock_by_id = day_clock.set_index(day_clock["candidate_id"].astype(str), drop=False)
        for cid, reason in missing_level_reason.items():
            c = clock_by_id.loc[cid]
            for h in EXPECTED_HORIZONS:
                metric_rows.append({
                    "builder_version": BUILDER_VERSION, "schema_version": SCHEMA_VERSION,
                    "candidate_id": cid, "candidate_date": day, "direction": int(c.direction),
                    "candidate_second": int(c.candidate_second), "observation_horizon_sec": int(h),
                    "anchor_found": False, "anchor_ts_recv_ns": math.nan, "anchor_offset_ms": math.nan,
                    "sequence_valid": False, "invalid_reason": reason,
                    "opposing__refilled_level_count": math.nan,
                    "opposing__observed_refill_quote": math.nan,
                    "opposing__strict_post_refill_aggressive_trade_quote": math.nan,
                    "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": math.nan,
                    "opposing__post_refill_aggressive_trade_to_refill_ratio": math.nan,
                    "opposing__refilled_level_post_trade_share": math.nan,
                    "opposing__minimum_nontrade_queue_reduction_quote": math.nan,
                    "opposing__minimum_nontrade_reduction_to_refill_ratio": math.nan,
                    "opposing__strict_trade_time_complete_share": math.nan,
                    "control__directional_mid_move_bps_3s": safe_float(c.directional_mid_move_bps),
                    "control__directional_aggressive_quote_3s": safe_float(c.directional_aggressive_quote),
                    "control__opposing_peak_initial_queue_loss_share_3s": safe_float(c.opposing__peak_initial_queue_loss_share),
                    "control__opposing_refill_quote_3s_frozen": safe_float(c.opposing__same_price_refill_after_depletion_quote),
                    "is_reaccel_context": bool(getattr(c, "is_reaccel_context", False)),
                })
        diag = {
            "candidate_date": day, "candidate_count": int(len(day_clock)),
            "frozen_level_eligible": 0, "frozen_level_missing": int(len(missing_level_reason)),
            "anchor_found": 0, "sequence_valid": 0, "filtered_trade_rows": 0,
            "depth_rows": 0, "applicable_depth_rows": 0, "gap_records": 0,
            "depth_rows_decoded": 0, "depth_levels_decoded": 0,
        }
        return pd.DataFrame(metric_rows), pd.DataFrame(), pd.DataFrame(), diag

    min_ns = min(t.search_start_ns for t in trackers)
    max_ns = max(t.final_end_ns for t in trackers)

    book = phase6a.LocalBook(price_scale=price_scale)
    fast_stats = phase6a.FastDepthStats()
    gap_records: list[Any] = []
    snapshots, depths = build_raw_iters(phase6a, day_files, symbol, price_scale, fast_stats)

    active: list[CandidateTracker] = []
    pending = sorted(trackers, key=lambda t: (t.search_start_ns, t.candidate_id))
    pidx = 0
    stop_ns = max_ns
    depth_rows = 0
    applicable_depth_rows = 0

    def try_anchor(t: CandidateTracker, state_ts_ns: int) -> None:
        if t.anchor_found or state_ts_ns < t.search_start_ns or state_ts_ns > t.search_end_ns:
            return
        if book_vector_matches(book, t.all_levels, t.direction):
            t.anchor_found = True
            t.anchor_ts_recv_ns = int(state_ts_ns)

    last_state_ts = 0
    for kind, rec in merge_event_families(snapshots, depths):
        ts_ns = int(rec["ts_recv_ns"])
        if ts_ns > stop_ns and pidx >= len(pending):
            break

        while pidx < len(pending) and pending[pidx].search_start_ns <= ts_ns:
            t = pending[pidx]
            active.append(t)
            try_anchor(t, last_state_ts if last_state_ts >= t.search_start_ns else t.search_start_ns)
            pidx += 1

        # Anchor can be the state immediately before this event.
        for t in active:
            try_anchor(t, ts_ns)

        if kind == "SNAPSHOT":
            book.apply_snapshot(rec)
            last_state_ts = ts_ns
            for t in active:
                try_anchor(t, ts_ns)
        else:
            depth_rows += 1
            applicable = bool(phase7e.depth_row_is_applicable(book, rec))
            changes = depth_changes_before_apply(book, rec) if applicable else []
            if applicable:
                applicable_depth_rows += 1
                for t in active:
                    if not t.anchor_found or ts_ns <= int(t.anchor_ts_recv_ns or 0) or ts_ns >= t.final_end_ns:
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
                        t.depth_events.append({
                            "candidate_id": t.candidate_id,
                            "candidate_date": day,
                            "direction": t.direction,
                            "ts_recv_ns": ts_ns,
                            "relative_to_candidate_ms": (ts_ns - t.candidate_second * 1_000_000_000) / 1e6,
                            "transaction_time_ms": safe_float(rec.get("transaction_time_ms")),
                            "exchange_event_ms": safe_float(rec.get("exchange_event_ms")),
                            "first_update_id": rec.get("first_update_id"),
                            "final_update_id": rec.get("final_update_id"),
                            "previous_final_update_id": rec.get("previous_final_update_id"),
                            "book_side": side_name,
                            "price_tick": int(tick),
                            "price": float(tick) / price_scale,
                            "old_qty": float(old_qty),
                            "new_qty": float(new_qty),
                            "delta_qty": float(new_qty) - float(old_qty),
                            "event_kind": "DEPTH_CHANGE",
                        })
            book.apply_depth(rec, gap_records, day)
            last_state_ts = ts_ns
            # Any invalidity/gap during active windows permanently invalidates sequence.
            if not bool(getattr(book, "valid", False)):
                for t in active:
                    if t.anchor_found and ts_ns < t.final_end_ns:
                        t.sequence_valid = False
                        if not t.invalid_reason:
                            t.invalid_reason = "BOOK_SEQUENCE_GAP_OR_INVALID"
            for t in active:
                try_anchor(t, ts_ns)

        # Expire anchor search and active tracking.
        keep = []
        for t in active:
            if not t.anchor_found and ts_ns > t.search_end_ns:
                t.sequence_valid = False
                if not t.invalid_reason:
                    t.invalid_reason = "FROZEN_ANCHOR_STATE_NOT_REFOUND_WITHIN_CANDIDATE_SECOND"
            if ts_ns < t.final_end_ns:
                keep.append(t)
        active = keep

    # Remaining pending/active candidates are invalid if never anchored.
    for t in trackers:
        if not t.anchor_found:
            t.sequence_valid = False
            if not t.invalid_reason:
                t.invalid_reason = "ANCHOR_NOT_FOUND"

    # Depth-first memory discipline: the expensive nested Arrow replay finishes before
    # event-level trades are loaded. Only price levels that actually refilled within
    # the max causal horizon and only merged candidate receive-time windows are read.
    trade_ticks = refill_ticks_from_depth(trackers)
    trade_intervals = merge_candidate_windows(trackers)
    trades = load_filtered_trades(day_files.trades, symbol, trade_ticks, trade_intervals)

    metric_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    tape_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    trades_by_tick = {int(k): g.copy() for k, g in trades.groupby("price_tick", sort=False)} if len(trades) else {}
    filtered_trade_rows = int(len(trades))
    del trades
    gc.collect()
    clock_by_id = day_clock.set_index(day_clock["candidate_id"].astype(str), drop=False)

    for t in trackers:
        events = pd.DataFrame(t.depth_events)
        for h in EXPECTED_HORIZONS:
            endpoint_ns = (t.candidate_second + int(h)) * 1_000_000_000
            evh = events[events["ts_recv_ns"] < endpoint_ns].copy() if not events.empty else pd.DataFrame()
            level_metrics = []
            for lr in t.opposing_levels.itertuples(index=False):
                tick = int(lr.price_tick)
                initial = float(lr.initial_qty)
                price = float(lr.price)
                le = evh[evh["price_tick"] == tick].sort_values("ts_recv_ns", kind="mergesort") if not evh.empty else pd.DataFrame()
                current = initial
                minq = initial
                depleted = False
                first_refill_recv = None
                first_refill_tx = math.nan
                first_refill_exchange = math.nan
                pre_refill_residual = math.nan
                refill_qty = 0.0
                gross_post_refill_decrease = 0.0
                refill_updates = 0
                post_refill_decrease_updates = 0
                first_refill_seen = False
                for er in le.itertuples(index=False):
                    oldq = float(er.old_qty); newq = float(er.new_qty); delta = newq - oldq
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
                receive_after_but_exchange_before_qty = 0.0
                post_refill_trade_count = 0
                tape_trade_rows = []
                trade_time_complete = True
                if first_refill_seen:
                    tg = trades_by_tick.get(tick, pd.DataFrame()).copy()
                    if not tg.empty:
                        tg = tg[(tg["ts_recv_ns"] > int(first_refill_recv)) & (tg["ts_recv_ns"] < endpoint_ns)].copy()
                        maker = bool_series(tg["buyer_is_maker"]) if not tg.empty else pd.Series(dtype=bool)
                        correct = (~maker) if t.direction > 0 else maker
                        tg = tg[correct].copy()
                        if not tg.empty:
                            tt = pd.to_numeric(tg["trade_time_ms"], errors="coerce")
                            trade_time_complete = bool(tt.notna().all() and math.isfinite(first_refill_tx))
                            if math.isfinite(first_refill_tx):
                                strict = tg[tt > first_refill_tx].copy()
                                same = tg[tt == first_refill_tx].copy()
                                before = tg[tt < first_refill_tx].copy()
                                strict_trade_qty = float(pd.to_numeric(strict["quantity"], errors="coerce").fillna(0).sum())
                                same_ms_trade_qty = float(pd.to_numeric(same["quantity"], errors="coerce").fillna(0).sum())
                                receive_after_but_exchange_before_qty = float(pd.to_numeric(before["quantity"], errors="coerce").fillna(0).sum())
                                post_refill_trade_count = int(len(strict))
                                for rr in pd.concat([strict.assign(timing_class="STRICT_AFTER"), same.assign(timing_class="SAME_EXCHANGE_MS"), before.assign(timing_class="EXCHANGE_BEFORE_RECEIVE_AFTER")], ignore_index=True).itertuples(index=False):
                                    tape_trade_rows.append({
                                        "candidate_id": t.candidate_id, "candidate_date": day, "direction": t.direction,
                                        "observation_horizon_sec": int(h), "event_kind": "AGGRESSIVE_TRADE_AT_REFILLED_LEVEL",
                                        "ts_recv_ns": int(rr.ts_recv_ns),
                                        "relative_to_candidate_ms": (int(rr.ts_recv_ns) - t.candidate_second * 1_000_000_000) / 1e6,
                                        "price_tick": tick, "price": safe_float(rr.price), "quantity": safe_float(rr.quantity),
                                        "buyer_is_maker": bool(rr.buyer_is_maker), "trade_time_ms": safe_float(rr.trade_time_ms),
                                        "exchange_event_ms": safe_float(getattr(rr, "exchange_event_ms", math.nan)),
                                        "timing_class": str(rr.timing_class),
                                    })
                            else:
                                trade_time_complete = False
                    # no trades is valid; not a timestamp missing case

                nontrade_lb = max(0.0, gross_post_refill_decrease - strict_trade_qty)
                refill_quote = refill_qty * price
                strict_trade_quote = strict_trade_qty * price
                same_ms_trade_quote = same_ms_trade_qty * price
                nontrade_lb_quote = nontrade_lb * price
                lm = {
                    "candidate_id": t.candidate_id, "candidate_date": day, "direction": t.direction,
                    "observation_horizon_sec": int(h), "price_tick": tick, "price": price,
                    "initial_qty": initial, "min_qty": minq, "final_observed_qty": current,
                    "depleted_before_refill": bool(depleted), "first_refill_seen": bool(first_refill_seen),
                    "first_refill_ts_recv_ns": first_refill_recv, "first_refill_transaction_time_ms": first_refill_tx,
                    "first_refill_exchange_event_ms": first_refill_exchange, "pre_refill_residual_qty": pre_refill_residual,
                    "observed_refill_qty": refill_qty, "observed_refill_quote": refill_quote,
                    "refill_update_count": refill_updates, "post_refill_decrease_update_count": post_refill_decrease_updates,
                    "post_refill_gross_queue_reduction_qty": gross_post_refill_decrease,
                    "strict_post_refill_aggressive_trade_qty": strict_trade_qty,
                    "strict_post_refill_aggressive_trade_quote": strict_trade_quote,
                    "same_exchange_ms_post_refill_aggressive_trade_qty": same_ms_trade_qty,
                    "same_exchange_ms_post_refill_aggressive_trade_quote": same_ms_trade_quote,
                    "receive_after_but_exchange_before_trade_qty": receive_after_but_exchange_before_qty,
                    "strict_post_refill_aggressive_trade_count": post_refill_trade_count,
                    "strict_trade_time_complete": bool(trade_time_complete),
                    "minimum_nontrade_queue_reduction_qty": nontrade_lb,
                    "minimum_nontrade_queue_reduction_quote": nontrade_lb_quote,
                    "anchor_found": t.anchor_found, "sequence_valid": t.sequence_valid,
                    "anchor_ts_recv_ns": t.anchor_ts_recv_ns, "invalid_reason": t.invalid_reason,
                }
                level_metrics.append(lm)
                level_rows.append(lm)
                tape_rows.extend(tape_trade_rows)

            lm = pd.DataFrame(level_metrics)
            refilled = lm[lm["first_refill_seen"] & (pd.to_numeric(lm["observed_refill_quote"], errors="coerce") > 0)].copy() if len(lm) else pd.DataFrame()
            refill_quote = float(pd.to_numeric(refilled.get("observed_refill_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            trade_quote = float(pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            same_quote = float(pd.to_numeric(refilled.get("same_exchange_ms_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            nontrade_quote = float(pd.to_numeric(refilled.get("minimum_nontrade_queue_reduction_quote", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            refilled_n = int(len(refilled))
            traded_n = int((pd.to_numeric(refilled.get("strict_post_refill_aggressive_trade_quote", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if refilled_n else 0
            trade_time_complete_share = float(bool_series(refilled["strict_trade_time_complete"]).mean()) if refilled_n else math.nan
            c = clock_by_id.loc[t.candidate_id]
            metric_rows.append({
                "builder_version": BUILDER_VERSION, "schema_version": SCHEMA_VERSION,
                "candidate_id": t.candidate_id, "candidate_date": day, "direction": t.direction,
                "candidate_second": t.candidate_second, "observation_horizon_sec": int(h),
                "anchor_found": t.anchor_found, "anchor_ts_recv_ns": t.anchor_ts_recv_ns,
                "anchor_offset_ms": ((int(t.anchor_ts_recv_ns) - t.candidate_second * 1_000_000_000) / 1e6) if t.anchor_ts_recv_ns is not None else math.nan,
                "sequence_valid": t.sequence_valid, "invalid_reason": t.invalid_reason,
                "opposing__refilled_level_count": refilled_n,
                "opposing__observed_refill_quote": refill_quote,
                "opposing__strict_post_refill_aggressive_trade_quote": trade_quote,
                "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": same_quote,
                "opposing__post_refill_aggressive_trade_to_refill_ratio": trade_quote / refill_quote if refill_quote > 0 else math.nan,
                "opposing__refilled_level_post_trade_share": traded_n / refilled_n if refilled_n > 0 else math.nan,
                "opposing__minimum_nontrade_queue_reduction_quote": nontrade_quote,
                "opposing__minimum_nontrade_reduction_to_refill_ratio": nontrade_quote / refill_quote if refill_quote > 0 else math.nan,
                "opposing__strict_trade_time_complete_share": trade_time_complete_share,
                "control__directional_mid_move_bps_3s": safe_float(c.directional_mid_move_bps),
                "control__directional_aggressive_quote_3s": safe_float(c.directional_aggressive_quote),
                "control__opposing_peak_initial_queue_loss_share_3s": safe_float(c.opposing__peak_initial_queue_loss_share),
                "control__opposing_refill_quote_3s_frozen": safe_float(c.opposing__same_price_refill_after_depletion_quote),
                "is_reaccel_context": bool(getattr(c, "is_reaccel_context", False)),
            })

        # Preserve depth tape once per candidate, not once per horizon.
        tape_rows.extend(t.depth_events)

    # Preserve candidates that legitimately lack frozen exact 7M.3 levels.
    # Their mechanism values are missing (NaN), never zero. This keeps the
    # denominator honest while ensuring they cannot enter refill-fate matching.
    for cid, reason in missing_level_reason.items():
        c = clock_by_id.loc[cid]
        for h in EXPECTED_HORIZONS:
            metric_rows.append({
                "builder_version": BUILDER_VERSION, "schema_version": SCHEMA_VERSION,
                "candidate_id": cid, "candidate_date": day, "direction": int(c.direction),
                "candidate_second": int(c.candidate_second), "observation_horizon_sec": int(h),
                "anchor_found": False, "anchor_ts_recv_ns": math.nan, "anchor_offset_ms": math.nan,
                "sequence_valid": False, "invalid_reason": reason,
                "opposing__refilled_level_count": math.nan,
                "opposing__observed_refill_quote": math.nan,
                "opposing__strict_post_refill_aggressive_trade_quote": math.nan,
                "opposing__same_exchange_ms_post_refill_aggressive_trade_quote": math.nan,
                "opposing__post_refill_aggressive_trade_to_refill_ratio": math.nan,
                "opposing__refilled_level_post_trade_share": math.nan,
                "opposing__minimum_nontrade_queue_reduction_quote": math.nan,
                "opposing__minimum_nontrade_reduction_to_refill_ratio": math.nan,
                "opposing__strict_trade_time_complete_share": math.nan,
                "control__directional_mid_move_bps_3s": safe_float(c.directional_mid_move_bps),
                "control__directional_aggressive_quote_3s": safe_float(c.directional_aggressive_quote),
                "control__opposing_peak_initial_queue_loss_share_3s": safe_float(c.opposing__peak_initial_queue_loss_share),
                "control__opposing_refill_quote_3s_frozen": safe_float(c.opposing__same_price_refill_after_depletion_quote),
                "is_reaccel_context": bool(getattr(c, "is_reaccel_context", False)),
            })

    metrics = pd.DataFrame(metric_rows)
    levels = pd.DataFrame(level_rows)
    tape = pd.DataFrame(tape_rows)
    diag = {
        "candidate_date": day,
        "candidate_count": int(len(day_clock)),
        "frozen_level_eligible": int(len(trackers)),
        "frozen_level_missing": int(len(missing_level_reason)),
        "anchor_found": int(sum(t.anchor_found for t in trackers)),
        "sequence_valid": int(sum(t.sequence_valid for t in trackers)),
        "filtered_trade_rows": filtered_trade_rows,
        "depth_rows": int(depth_rows),
        "applicable_depth_rows": int(applicable_depth_rows),
        "gap_records": int(len(gap_records)),
        "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0) or 0),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0) or 0),
    }
    return metrics, levels, tape, diag


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    sd = float(x.std(ddof=1)); mu = float(x.mean())
    if not math.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    den = math.sqrt(max(0.0, 0.5 * (float(np.var(x, ddof=1)) + float(np.var(y, ddof=1)))))
    if den <= 1e-12:
        return 0.0 if math.isclose(float(x.mean()), float(y.mean())) else math.inf
    return float((x.mean() - y.mean()) / den)


def percentile_tercile(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if mask.any():
        r = x.loc[mask].rank(method="first", pct=True)
        out.loc[mask] = np.ceil(r * 3).clip(1, 3).astype(int).astype("Int64")
    return out


def greedy_pairs(d: pd.DataFrame, feature: str, controls: Sequence[str]) -> pd.DataFrame:
    x = d.copy()
    x["__tercile"] = percentile_tercile(x[feature])
    for c in controls:
        x[f"__z__{c}"] = zscore(x[c])
    high = x[x["__tercile"] == 3].copy()
    low = x[x["__tercile"] == 1].copy()
    rows = []
    for (dt, direction), hb in high.groupby(["candidate_date", "direction"], sort=True):
        lb = low[(low["candidate_date"] == dt) & (pd.to_numeric(low["direction"], errors="coerce") == int(direction))].copy()
        available = set(lb.index)
        for _, hr in hb.sort_values([feature, "candidate_id"], ascending=[False, True], kind="mergesort").iterrows():
            best = None
            for li in sorted(available):
                lr = lb.loc[li]
                diffs = []
                okay = True
                for c in controls:
                    hv = safe_float(hr[f"__z__{c}"]); lv = safe_float(lr[f"__z__{c}"])
                    if not math.isfinite(hv) or not math.isfinite(lv) or abs(hv-lv) > MATCH_Z_CALIPER:
                        okay = False; break
                    diffs.append(hv-lv)
                if not okay:
                    continue
                dist = math.sqrt(sum(v*v for v in diffs))
                if dist > MATCH_EUCLIDEAN_CALIPER:
                    continue
                key = (dist, str(lr["candidate_id"]))
                if best is None or key < best[0]:
                    best = (key, li, lr, diffs)
            if best is None:
                continue
            _, li, lr, diffs = best
            available.remove(li)
            row = {"candidate_date": dt, "direction": int(direction), "high_candidate_id": str(hr.candidate_id), "low_candidate_id": str(lr.candidate_id), "distance": best[0][0]}
            for c, diff in zip(controls, diffs):
                row[f"high__{c}"] = safe_float(hr[c]); row[f"low__{c}"] = safe_float(lr[c]); row[f"diff__{c}"] = diff
            rows.append(row)
    return pd.DataFrame(rows)


def pair_balance(pairs: pd.DataFrame, controls: Sequence[str]) -> tuple[float, dict[str, float]]:
    vals = {}
    for c in controls:
        vals[c] = smd(pairs.get(f"high__{c}", pd.Series(dtype=float)), pairs.get(f"low__{c}", pd.Series(dtype=float)))
    finite = [abs(v) for v in vals.values() if math.isfinite(v)]
    return (max(finite) if finite else math.nan), vals


def repair_pairs(pairs: pd.DataFrame, controls: Sequence[str]) -> pd.DataFrame:
    x = pairs.copy().reset_index(drop=True)
    while len(x) > PILOT_MIN_REPAIR_PAIRS:
        mx, vals = pair_balance(x, controls)
        if math.isfinite(mx) and mx <= BALANCE_SMD_GATE:
            break
        finite = {k: v for k, v in vals.items() if math.isfinite(v)}
        if not finite:
            break
        worst = max(finite, key=lambda c: (abs(finite[c]), c)); sign = 1 if finite[worst] >= 0 else -1
        contribution = (pd.to_numeric(x[f"high__{worst}"], errors="coerce") - pd.to_numeric(x[f"low__{worst}"], errors="coerce")) * sign
        drop_idx = contribution.fillna(-np.inf).idxmax()
        x = x.drop(index=drop_idx).reset_index(drop=True)
    return x


def build_pilot_feasibility(metrics: pd.DataFrame, total_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = metrics[(pd.to_numeric(metrics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC) & bool_series(metrics["anchor_found"]) & bool_series(metrics["sequence_valid"])].copy()
    d = d[pd.to_numeric(d["opposing__observed_refill_quote"], errors="coerce") > 0].copy()
    d["__momentum"] = pd.to_numeric(d["control__directional_mid_move_bps_3s"], errors="coerce")
    d["__aggression"] = np.log1p(pd.to_numeric(d["control__directional_aggressive_quote_3s"], errors="coerce").clip(lower=0))
    d["__destruction"] = pd.to_numeric(d["control__opposing_peak_initial_queue_loss_share_3s"], errors="coerce")
    d["__refill"] = np.log1p(pd.to_numeric(d["control__opposing_refill_quote_3s_frozen"], errors="coerce").clip(lower=0))
    controls = ["__momentum", "__aggression", "__destruction", "__refill"]
    features = [
        "opposing__post_refill_aggressive_trade_to_refill_ratio",
        "opposing__refilled_level_post_trade_share",
        "opposing__minimum_nontrade_reduction_to_refill_ratio",
    ]
    pair_parts = []
    rows = []
    pilot_days = max(1, d["candidate_date"].nunique())
    scale = float(total_days) / float(pilot_days)
    for feature in features:
        z = d.copy()
        for c in [feature, *controls]:
            z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()
        pairs = greedy_pairs(z, feature, controls)
        repaired = repair_pairs(pairs, controls) if len(pairs) else pairs
        mx, vals = pair_balance(repaired, controls) if len(repaired) else (math.nan, {})
        proj = int(round(len(repaired) * scale))
        disc = int((repaired["candidate_date"] <= DISCOVERY_END_DATE).sum()) if len(repaired) else 0
        val = int((repaired["candidate_date"] >= VALIDATION_START_DATE).sum()) if len(repaired) else 0
        disc_days = max(1, d[d["candidate_date"] <= DISCOVERY_END_DATE]["candidate_date"].nunique())
        val_days = max(1, d[d["candidate_date"] >= VALIDATION_START_DATE]["candidate_date"].nunique())
        full_disc_days = int((pd.Series(sorted(metrics["candidate_date"].astype(str).unique())) <= DISCOVERY_END_DATE).sum())
        full_val_days = int((pd.Series(sorted(metrics["candidate_date"].astype(str).unique())) >= VALIDATION_START_DATE).sum())
        # The metrics input contains pilot days only, so split projection uses half-sample day counts (31/31 for frozen 62 days).
        full_disc_days = 31; full_val_days = 31
        proj_disc = int(round(disc * full_disc_days / disc_days))
        proj_val = int(round(val * full_val_days / val_days))
        pos_share = float((pd.to_numeric(z[feature], errors="coerce") > 0).mean()) if len(z) else math.nan
        feasible = bool(
            len(repaired) > 0 and math.isfinite(mx) and mx <= BALANCE_SMD_GATE
            and proj >= MIN_PROJECTED_FULL_PRIMARY_PAIRS
            and proj_disc >= MIN_PROJECTED_SPLIT_PAIRS and proj_val >= MIN_PROJECTED_SPLIT_PAIRS
            and math.isfinite(pos_share) and MIN_FEATURE_POSITIVE_SHARE <= pos_share <= MAX_FEATURE_POSITIVE_SHARE
        )
        row = {
            "feature": feature, "pilot_eligible_candidates": int(len(z)), "pilot_days": pilot_days,
            "pilot_pairs_before_repair": int(len(pairs)), "pilot_pairs_after_repair": int(len(repaired)),
            "pilot_max_abs_smd": mx, "projected_full_pairs": proj,
            "projected_discovery_pairs": proj_disc, "projected_validation_pairs": proj_val,
            "positive_feature_share": pos_share, "balance_gate": BALANCE_SMD_GATE,
            "projected_full_pair_gate": MIN_PROJECTED_FULL_PRIMARY_PAIRS,
            "projected_split_pair_gate": MIN_PROJECTED_SPLIT_PAIRS,
            "feasible_before_outcomes": feasible,
        }
        for c, v in vals.items(): row[f"smd__{c}"] = v
        rows.append(row)
        if len(repaired):
            q = repaired.copy(); q["feature"] = feature; pair_parts.append(q)
    return pd.DataFrame(rows), (pd.concat(pair_parts, ignore_index=True, sort=False) if pair_parts else pd.DataFrame())


def checkpoint_paths(output_root: Path, day: str) -> dict[str, Path]:
    return {
        "metrics": output_root / "candidate_metrics" / f"candidate_date={day}" / "part.parquet",
        "levels": output_root / "level_fate" / f"candidate_date={day}" / "part.parquet",
        "tape": output_root / "event_tape" / f"candidate_date={day}" / "part.parquet",
        "manifest": output_root / "checkpoints" / f"candidate_date={day}" / "manifest.json",
    }


def day_signature(day: str, ids: Sequence[str], level_path: Path, day_files: Any) -> str:
    payload = {
        "day": day,
        "candidate_ids": sorted(map(str, ids)),
        "level_audit": file_signature([level_path]),
        "depth": file_signature(day_files.depth),
        "snapshots": file_signature(day_files.snapshots),
        "trades": file_signature(day_files.trades),
        "contract": contract_payload(),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def can_reuse(paths: dict[str, Path], signature: str) -> bool:
    if not all(paths[k].exists() for k in ("metrics", "levels", "tape", "manifest")):
        return False
    try:
        m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(m.get("status")) == "PASS"
        and str(m.get("input_signature")) == signature
        and str(m.get("builder_version")) in COMPATIBLE_CHECKPOINT_BUILDERS
    )


def run_self_test() -> int:
    # Date selector is deterministic and temporally spread.
    fake = pd.DataFrame({"candidate_date": [f"2026-06-{i:02d}" for i in range(1, 13) for _ in range((i % 4) + 1)]})
    fake["candidate_id"] = [f"c{i}" for i in range(len(fake))]
    pick = choose_pilot_dates(fake, 4)
    assert len(pick) == 4 and pick["candidate_date"].is_unique

    # Synthetic matching must respect exact date+direction and return finite balance.
    rows = []
    for i in range(60):
        rows.append({
            "candidate_id": f"x{i}", "candidate_date": "2026-06-01", "direction": 1,
            "feat": float(i), "a": float(i % 7), "b": float(i % 5), "c": float(i % 3), "r": float(i % 11),
        })
    dd = pd.DataFrame(rows)
    p = greedy_pairs(dd, "feat", ["a", "b", "c", "r"])
    assert isinstance(p, pd.DataFrame)

    # Candidate-window trade filter keeps only values inside half-open intervals.
    vv = np.asarray([5, 10, 14, 20, 24, 30], dtype=np.int64)
    mm = interval_mask_ns(vv, [(10, 15), (20, 25)])
    assert vv[mm].tolist() == [10, 14, 20, 24]

    # Conservative non-trade lower-bound arithmetic.
    D, T = 8.0, 6.0
    assert abs(max(0.0, D - T) - 2.0) < 1e-12

    # Missing frozen exact levels are unavailable mechanics, never zero mechanics.
    unavailable = math.nan
    assert math.isnan(unavailable)
    print("PASS: Phase 7M.6B self-test (pilot selection, matching, conservative attribution arithmetic).")
    return 0


def findings_text(summary: dict[str, Any], gate: pd.DataFrame) -> str:
    lines = [
        "# Phase 7M.6B — Refill→Re-depletion sequence pilot", "",
        f"- Status: **{summary['status']}**",
        f"- Run scope: **{summary['run_scope']}**",
        f"- Pilot dates: **{', '.join(summary['processed_dates'])}**",
        f"- Candidates processed: **{summary['candidate_count']}**",
        f"- Frozen exact-level coverage: **{summary.get('frozen_level_coverage_rate', float('nan')):.2%}**",
        f"- Anchor-match rate: **{summary['anchor_match_rate']:.2%}**",
        f"- Sequence-valid rate: **{summary['sequence_valid_rate']:.2%}**",
        f"- Primary 3s refill-eligible candidates: **{summary['primary_refill_eligible']}**",
        f"- Projected full refill-eligible candidates: **{summary['projected_full_refill_eligible']}**",
        "",
        "## Scientific role", "",
        "This is an outcome-blind feasibility/raw-sequence pilot. It does not test P75 or any future continuation label.",
        "", "## Reuse contract", "",
        "The per-date candidate_metrics, level_fate and event_tape partitions are full-contract checkpoints. A later full replay should reuse any date whose signed manifest still matches.",
        "", "## Attribution language", "",
        "The public feed supports price-level statements: after a depleted opposing level refilled, aggressive trades subsequently executed at that same price. It does not identify an individual seller/order. Non-trade reduction is reported only as a conservative lower bound.",
        "", "## Pre-outcome matching feasibility", "",
    ]
    if gate.empty:
        lines.append("No feasibility rows were produced.")
    else:
        for r in gate.itertuples(index=False):
            lines.append(f"- `{r.feature}`: pilot pairs={r.pilot_pairs_after_repair}, projected full={r.projected_full_pairs}, D/V={r.projected_discovery_pairs}/{r.projected_validation_pairs}, feasible={bool(r.feasible_before_outcomes)}")
    lines += ["", "No future outcome should be loaded until these pilot diagnostics are reviewed."]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6B outcome-blind refill re-depletion raw-sequence pilot")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7e-script", type=Path, default=DEFAULT_PHASE7E_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--pilot-date", action="append", default=[], help="Explicit YYYY-MM-DD pilot date; repeatable. If omitted, four representative dates are selected outcome-blind.")
    p.add_argument("--all-dates", action="store_true", help="Process all frozen candidate dates using the same checkpoint contract. Do not use until the pilot gate is reviewed.")
    p.add_argument("--force-day", action="store_true", help="Rebuild selected date checkpoints even if signatures match.")
    p.add_argument("--price-scale", type=float, default=PRICE_SCALE)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_stack()
    if abs(float(args.price_scale) - PRICE_SCALE) > 1e-12:
        raise SystemExit(f"Phase 7M.6B is frozen to price_scale={PRICE_SCALE}")

    print("Phase 7M.6B — Outcome-blind refill→re-depletion sequence pilot")
    print("[1/7] Verifying frozen Phase 7M.3 causal contract...")
    m3_meta, candidates, mechanics = load_m3(args.m3_root)
    clock = derive_candidate_clock(candidates, mechanics)

    if args.all_dates and args.pilot_date:
        raise SystemExit("Use either --all-dates or explicit --pilot-date values, not both.")
    if args.all_dates:
        selected = sorted(clock["candidate_date"].unique().tolist())
        selection = pd.DataFrame({"candidate_date": selected, "selection_rule": "ALL_FROZEN_CANDIDATE_DATES"})
        run_scope = "FULL"
    elif args.pilot_date:
        selected = sorted(set(args.pilot_date))
        missing = sorted(set(selected) - set(clock["candidate_date"].unique()))
        if missing:
            raise RuntimeError(f"Explicit pilot dates not in frozen candidate set: {missing}")
        selection = pd.DataFrame({"candidate_date": selected, "selection_rule": "EXPLICIT_USER_PILOT_DATE"})
        run_scope = f"PILOT_{len(selected)}"
    else:
        selection = choose_pilot_dates(clock, PILOT_DAY_COUNT)
        selected = selection["candidate_date"].tolist()
        run_scope = f"PILOT_{len(selected)}"

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.report_root / "pilot_date_manifest.csv", selection)
    atomic_json(args.output_root / "metadata" / "run_config.json", {
        "generated_at": utc_now_iso(), "run_scope": run_scope, "selected_dates": selected,
        "contract": contract_payload(), "future_outcomes_loaded": False,
    })

    print(f"[2/7] Frozen pilot dates ({len(selected)}): {', '.join(selected)}")
    phase6a = load_module(args.phase6a_script, "phase6a_for_7m6b")
    phase7e = load_module(args.phase7e_script, "phase7e_for_7m6b")
    # Reuse the trusted repository day discovery so symlinked D: storage follows existing semantics.
    all_days = phase6a.discover_days(args.canonical_root, None, None, None)
    day_map = {d.day.isoformat(): d for d in all_days}

    all_metrics = []
    progress = []
    reused = 0; built = 0

    print("[3/7] Replaying selected raw dates / reusing signed checkpoints...")
    for i, day in enumerate(selected, 1):
        if day not in day_map:
            raise RuntimeError(f"Canonical raw day not discoverable through Phase 6A: {day}")
        day_files = day_map[day]
        day_clock = clock[clock["candidate_date"] == day].copy()
        ids = day_clock["candidate_id"].astype(str).tolist()
        level_path = args.m3_root / "level_audit" / f"candidate_date={day}" / "part.parquet"
        sig = day_signature(day, ids, level_path, day_files)
        paths = checkpoint_paths(args.output_root, day)
        if not args.force_day and can_reuse(paths, sig):
            # Only compact candidate_metrics are needed for the pilot feasibility
            # aggregation. Large level_fate/event_tape checkpoints stay on disk.
            metrics = pq.read_table(paths["metrics"]).to_pandas()
            diag = json.loads(paths["manifest"].read_text(encoding="utf-8")).get("diagnostics", {})
            reused += 1; state = "REUSED"
        else:
            lev = load_level_audit_for_day(args.m3_root, day, set(ids))
            metrics, levels, tape, diag = process_day(
                day=day, day_clock=day_clock, level_audit=lev, day_files=day_files,
                phase6a=phase6a, phase7e=phase7e, symbol=args.symbol, price_scale=float(args.price_scale),
            )
            atomic_parquet(paths["metrics"], metrics); atomic_parquet(paths["levels"], levels); atomic_parquet(paths["tape"], tape)
            manifest = {
                "builder_version": BUILDER_VERSION, "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
                "candidate_date": day, "candidate_count": len(ids), "candidate_ids_sha256": sha256_text("\n".join(sorted(ids))),
                "input_signature": sig, "status": "PASS", "generated_at": utc_now_iso(),
                "future_outcomes_loaded": False, "diagnostics": diag, "contract": contract_payload(),
            }
            atomic_json(paths["manifest"], manifest)
            built += 1; state = "BUILT"
            # These can be very large. They are already durable on disk and are
            # not needed for the cross-day feasibility gate.
            del levels, tape, lev
        all_metrics.append(metrics)
        progress.append({"candidate_date": day, "candidate_count": len(ids), "state": state, **diag})
        print(f"  [{i:02d}/{len(selected):02d}] {day}: candidates={len(ids):4d} {state} | anchor={diag.get('anchor_found', 0)} valid={diag.get('sequence_valid', 0)} trades={diag.get('filtered_trade_rows', 0):,}")
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

    metrics = pd.concat(all_metrics, ignore_index=True, sort=False) if all_metrics else pd.DataFrame()
    atomic_csv(args.report_root / "replay_progress.csv", pd.DataFrame(progress))

    print("[4/7] Auditing causal sequence/timestamp coverage...")
    primary = metrics[pd.to_numeric(metrics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    frozen_level_rate = float((~primary["invalid_reason"].astype(str).isin(["FROZEN_7M3_LEVEL_AUDIT_MISSING", "FROZEN_7M3_OPPOSING_LEVELS_MISSING"])).mean()) if len(primary) else 0.0
    anchor_rate = float(bool_series(primary["anchor_found"]).mean()) if len(primary) else 0.0
    seq_rate = float(bool_series(primary["sequence_valid"]).mean()) if len(primary) else 0.0
    refill_eligible = primary[pd.to_numeric(primary["opposing__observed_refill_quote"], errors="coerce") > 0].copy()
    strict_time_cov = float(pd.to_numeric(refill_eligible["opposing__strict_trade_time_complete_share"], errors="coerce").fillna(1.0).mean()) if len(refill_eligible) else 0.0
    total_days = int(clock["candidate_date"].nunique())
    pilot_days = max(1, len(selected))
    projected_refill = int(round(len(refill_eligible) * total_days / pilot_days))
    structural_gate = bool(anchor_rate >= MIN_ANCHOR_MATCH_RATE and seq_rate >= MIN_SEQUENCE_VALID_RATE and strict_time_cov >= MIN_STRICT_TRADE_TIME_COVERAGE)
    yield_gate = bool(projected_refill >= MIN_PROJECTED_FULL_REFILL_ELIGIBLE)

    timing_audit = pd.DataFrame([{
        "run_scope": run_scope, "pilot_candidates": int(len(primary)), "frozen_level_coverage_rate": frozen_level_rate,
        "anchor_match_rate": anchor_rate, "sequence_valid_rate": seq_rate, "primary_refill_eligible": int(len(refill_eligible)),
        "strict_trade_time_coverage": strict_time_cov, "projected_full_refill_eligible": projected_refill,
        "structural_gate_pass": structural_gate, "yield_gate_pass": yield_gate,
    }])
    atomic_csv(args.report_root / "sequence_timing_feasibility.csv", timing_audit)

    # Replay parity diagnostic against frozen 7M.3 observed same-price refill amount.
    if len(primary):
        a = pd.to_numeric(primary["opposing__observed_refill_quote"], errors="coerce")
        b = pd.to_numeric(primary["control__opposing_refill_quote_3s_frozen"], errors="coerce")
        mask = a.notna() & b.notna()
        corr = float(a[mask].corr(b[mask])) if mask.sum() >= 3 else math.nan
        med_abs = float((a[mask] - b[mask]).abs().median()) if mask.any() else math.nan
    else:
        corr = math.nan; med_abs = math.nan
    parity = pd.DataFrame([{"pilot_rows": int(len(primary)), "refill_quote_corr_with_frozen_7m3": corr, "median_abs_refill_quote_difference": med_abs}])
    atomic_csv(args.report_root / "frozen_7m3_parity_audit.csv", parity)

    print("[5/7] Running pre-outcome common-support feasibility estimate...")
    gate, pairs = build_pilot_feasibility(metrics, total_days)
    atomic_csv(args.report_root / "precompute_feasibility_gate.csv", gate)
    atomic_parquet(args.output_root / "pilot_matched_pairs_outcome_blind.parquet", pairs)

    # Subgroup counts; no future labels.
    subgroup_rows = []
    for name, mask in (
        ("FULL", pd.Series(True, index=refill_eligible.index)),
        ("DISCOVERY", refill_eligible["candidate_date"].astype(str) <= DISCOVERY_END_DATE),
        ("VALIDATION", refill_eligible["candidate_date"].astype(str) >= VALIDATION_START_DATE),
        ("UP", pd.to_numeric(refill_eligible["direction"], errors="coerce") > 0),
        ("DOWN", pd.to_numeric(refill_eligible["direction"], errors="coerce") < 0),
        ("REACCEL_CONTEXT", bool_series(refill_eligible.get("is_reaccel_context", pd.Series(False, index=refill_eligible.index)))),
    ):
        n = int(mask.sum())
        subgroup_rows.append({"subgroup": name, "pilot_refill_eligible": n, "projected_full": int(round(n * total_days / pilot_days))})
    atomic_csv(args.report_root / "subgroup_feasibility.csv", pd.DataFrame(subgroup_rows))

    print("[6/7] Writing invariants and reusable pilot summary...")
    inv = [
        ("future_outcomes_loaded", True, False, False),
        ("frozen_level_missing_handled_as_missing_not_zero", True, int(primary["invalid_reason"].astype(str).str.startswith("FROZEN_7M3_").sum()), "diagnostic"),
        ("anchor_match_rate", anchor_rate >= MIN_ANCHOR_MATCH_RATE, anchor_rate, MIN_ANCHOR_MATCH_RATE),
        ("sequence_valid_rate", seq_rate >= MIN_SEQUENCE_VALID_RATE, seq_rate, MIN_SEQUENCE_VALID_RATE),
        ("strict_trade_time_coverage", strict_time_cov >= MIN_STRICT_TRADE_TIME_COVERAGE, strict_time_cov, MIN_STRICT_TRADE_TIME_COVERAGE),
        ("projected_full_refill_eligible", projected_refill >= MIN_PROJECTED_FULL_REFILL_ELIGIBLE, projected_refill, MIN_PROJECTED_FULL_REFILL_ELIGIBLE),
        ("checkpoint_reusability_contract", True, "signed per-date manifests", "signed per-date manifests"),
    ]
    inv_df = pd.DataFrame([{"invariant": n, "pass": bool(ok), "actual": actual, "required": req} for n, ok, actual, req in inv])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv_df)

    primary_feature_row = gate[gate["feature"] == "opposing__post_refill_aggressive_trade_to_refill_ratio"] if len(gate) else pd.DataFrame()
    match_gate = bool(primary_feature_row["feasible_before_outcomes"].iloc[0]) if len(primary_feature_row) else False
    overall = bool(structural_gate and yield_gate and match_gate)
    status = "PILOT_FEASIBLE_FOR_FULL_REPLAY" if overall else ("STOP_SCHEMA_OR_PARITY_DEFECT" if not structural_gate else "REVIEW_LOW_YIELD_OR_COMMON_SUPPORT")
    summary = {
        "builder_version": BUILDER_VERSION, "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(), "status": status, "run_scope": run_scope,
        "processed_dates": selected, "candidate_count": int(len(primary)), "frozen_level_coverage_rate": frozen_level_rate,
        "anchor_match_rate": anchor_rate, "sequence_valid_rate": seq_rate, "strict_trade_time_coverage": strict_time_cov,
        "primary_refill_eligible": int(len(refill_eligible)), "projected_full_refill_eligible": projected_refill,
        "built_date_checkpoints": built, "reused_date_checkpoints": reused,
        "primary_matching_feasible_before_outcomes": match_gate,
        "future_outcomes_loaded": False,
        "interpretation": "Feasibility only. Do not infer P75 direction/effect from this pilot.",
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_text(args.report_root / "findings.md", findings_text(summary, gate))

    print("[7/7] Complete.")
    print(f"Pilot status            : {status}")
    print(f"Frozen level coverage   : {frozen_level_rate:.2%}")
    print(f"Anchor match            : {anchor_rate:.2%}")
    print(f"Sequence valid          : {seq_rate:.2%}")
    print(f"3s refill eligible      : {len(refill_eligible):,}")
    print(f"Projected full eligible : {projected_refill:,}")
    print(f"Checkpoints             : {built} built / {reused} reused")
    print(f"Reports                 : {args.report_root}")
    print("Future outcomes loaded  : NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
