#!/usr/bin/env python3
"""Phase 7I.4 — causal all-market perps participant-interaction materialization.

Scientific role
---------------
Phase 7I.1/7I.2 froze a causal 35-bp candidate ontology:

    10-bp directional-change confirmation
        -> freeze the already-observed causal origin
        -> first 35-bp passage from that origin
        -> one candidate per causal episode

The audited full catalog contains 2,729 candidates (1,339 UP / 1,390 DOWN)
across 62 candidate dates and captures essentially all retrospective 75+/100+
physical moves.  The user approved that population for the expensive replay.

This builder DOES NOT regenerate, tune, filter, or relabel the candidate
universe.  It validates the frozen Phase-7I.1 candidate manifest and carries the
same candidate IDs into a reusable causal participant-interaction substrate.

For each candidate it reconstructs the authoritative Binance USD-M book with
the frozen Phase-6A contract and freezes the exact <=5-bp SUPPORT and OPPOSING
price levels at the causal 35-bp candidate second.  Those exact prices are
followed for 1/3/5/10 seconds.  Executed trades are first-class measurements in
the same windows.

Causal mechanics output (candidate_mechanics.parquet) contains only information
observable by each decision timestamp.  Future outcomes are isolated in a
separate candidate_decision_outcomes.parquet dataset.

Measurements
------------
PRICE / DECISION STATE
* causal origin and candidate price/displacement
* actual decision second (candidate + 1/3/5/10s)
* exact reconstructed anchor/end mid and directional mid movement
* actual last-trade decision price/displacement and threshold overshoot

EXECUTED TRADES
* directional/counter aggressive quote and counts
* aggressive imbalance
* total/average/max trade quote
* trade-size HHI and largest-trade share
* pre-window vs observation-window flow acceleration

EXACT BOOK RESPONSE
* exact same-price queue depletion
* retained initial liquidity
* durable unrecovered loss
* same-price refill and refill durability/latency
* near-touch restoration and outward-addition proxies
* support/opposing asymmetry

INTERACTION PRIMITIVES
* queue depletion relative to directional aggressive notional
* unrecovered loss per $1m directional aggression
* refill/restoration under directional pressure
* directional price movement per $1m net aggressive flow

FUTURE OUTCOMES (SEPARATE DATASET ONLY)
For each actual 1/3/5/10-second decision timestamp:
* origin completion to 50/75/100 bps over 5/10/15/30 minutes
* MFE/MAE from the actual decision price
* first +10/+20/+40 bps and -10/-15/-20 bps from the decision price
* time to those passages when observed

Important interpretation limits
--------------------------------
* Candidate generation is causal, but this remains historical all-market
  evaluation over dates already studied.  It is not an untouched forward holdout.
* A depth decrease is an exact observed quantity decrease, not an identified
  cancellation.  Same-price refill does not identify participant/order identity.
* buyer_is_maker is used only to infer aggressive trade direction in the same
  way as frozen Phase 6A.
* No classifier, optimized signal threshold, alpha, or profitability claim is
  produced here.  This phase materializes the reusable substrate for cheap
  downstream continuation-probability and economic analyses.

Runtime architecture
--------------------
This is intentionally the one expensive all-market candidate replay.  Work is
checkpointed atomically by candidate_date.  Completed date partitions are reused
only when the frozen candidate rows, raw canonical files, execution-price files,
and scientific configuration signatures match.  Large frozen-level audit tables
are persisted per date and summarized incrementally rather than retained across
the full run.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = 1
SCHEMA_NAME = "causal_candidate_participant_interaction_v1"
DEFAULT_SYMBOL = "BTCUSDT"

# Frozen upstream contracts.
REQUIRED_PHASE6A_VERSION = "1.0.4"
REQUIRED_PHASE7A_VERSION = "1.0.0"
REQUIRED_PHASE7E_VERSION = "1.0.3"
REQUIRED_PHASE7I1_VERSION = "1.0.0"
EXPECTED_FULL_CANDIDATES = 2_729
EXPECTED_FULL_UP = 1_339
EXPECTED_FULL_DOWN = 1_390
EXPECTED_FULL_CANDIDATE_DAYS = 62

PRICE_SCALE = 10.0
MAX_BOOK_AGE_MS = 2_000
FIXED_HORIZONS_SEC = (1, 3, 5, 10)
PRIMARY_HORIZON_SEC = 3
FROZEN_NEAR_BPS = 5.0
OUTER_BAND_BPS = 10.0
MAX_TRADE_STALE_SEC = 5
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
FAVORABLE_PASSAGES_BPS = (10.0, 20.0, 40.0)
ADVERSE_PASSAGES_BPS = (10.0, 15.0, 20.0)
MIN_PRIMARY_BOOK_COVERAGE = 0.95

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07a_build_large_move_event_catalog.py"
DEFAULT_PHASE7E_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07e_build_raw_depth_continuation_mechanics.py"
DEFAULT_PHASE7I1_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07i1_build_causal_large_move_candidate_catalog.py"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_CANDIDATE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "causal_large_move_candidate_catalog_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_causal_candidate_participant_interaction"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Phase 7I.4 requires numpy and pandas. Activate the project .venv first.") from exc
    return np, pd


def require_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("Phase 7I.4 Parquet I/O requires pyarrow. Activate the project .venv first.") from exc
    return np, pd, pa, pq


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7i4-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7i4-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_csv_atomic(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7i4-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_modules(args):
    p6 = load_module(args.phase6a_script, "phase6a_frozen_for_7i4")
    p7a = load_module(args.phase7a_script, "phase7a_frozen_for_7i4")
    p7e = load_module(args.phase7e_script, "phase7e_frozen_for_7i4")
    p7i1 = load_module(args.phase7i1_script, "phase7i1_frozen_for_7i4")
    found = {
        "Phase-6A": getattr(p6, "BUILDER_VERSION", None),
        "Phase-7A": getattr(p7a, "BUILDER_VERSION", None),
        "Phase-7E": getattr(p7e, "BUILDER_VERSION", None),
        "Phase-7I.1": getattr(p7i1, "BUILDER_VERSION", None),
    }
    expected = {
        "Phase-6A": REQUIRED_PHASE6A_VERSION,
        "Phase-7A": REQUIRED_PHASE7A_VERSION,
        "Phase-7E": REQUIRED_PHASE7E_VERSION,
        "Phase-7I.1": REQUIRED_PHASE7I1_VERSION,
    }
    bad = {k: (found[k], expected[k]) for k in expected if str(found[k]) != str(expected[k])}
    if bad:
        raise RuntimeError(f"Frozen builder-version mismatch: {bad}")
    return p6, p7a, p7e, p7i1


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted({Path(x).resolve() for x in paths}, key=lambda x: str(x)):
        st = p.stat()
        out.append({"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return out


def candidate_id_hash(candidates) -> str:
    ids = sorted(candidates["candidate_id"].astype(str).tolist())
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_frozen_candidates(args):
    _, pd, _, pq = require_stack()
    cp = args.candidate_root / "candidates.parquet"
    op = args.candidate_root / "candidate_outcomes.parquet"
    sp = args.candidate_root / "metadata" / "latest_summary.json"
    mp = args.candidate_root / "metadata" / "candidate_freeze_manifest.json"
    for p in (cp, op, sp, mp):
        if not p.exists():
            raise RuntimeError(f"Missing frozen Phase-7I.1 artifact: {p}")
    candidates = pq.read_table(cp).to_pandas()
    upstream_outcomes = pq.read_table(op).to_pandas()
    summary = json.loads(sp.read_text(encoding="utf-8"))
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if str(summary.get("builder_version")) != REQUIRED_PHASE7I1_VERSION or str(summary.get("status")) != "PASS":
        raise RuntimeError(f"Phase-7I.1 summary is not frozen PASS v{REQUIRED_PHASE7I1_VERSION}: {summary}")
    if len(candidates) != EXPECTED_FULL_CANDIDATES:
        raise RuntimeError(f"Approved Phase-7I.1 population requires {EXPECTED_FULL_CANDIDATES:,} candidates; found {len(candidates):,}")
    up = int((pd.to_numeric(candidates["direction"], errors="coerce") > 0).sum())
    down = int((pd.to_numeric(candidates["direction"], errors="coerce") < 0).sum())
    days = int(candidates["candidate_date"].astype(str).nunique())
    if (up, down, days) != (EXPECTED_FULL_UP, EXPECTED_FULL_DOWN, EXPECTED_FULL_CANDIDATE_DAYS):
        raise RuntimeError(
            "Approved Phase-7I.1 population geometry mismatch: "
            f"observed UP={up} DOWN={down} days={days}; expected "
            f"UP={EXPECTED_FULL_UP} DOWN={EXPECTED_FULL_DOWN} days={EXPECTED_FULL_CANDIDATE_DAYS}"
        )
    h = candidate_id_hash(candidates)
    mh = str(manifest.get("candidate_id_set_sha256", ""))
    sh = str(summary.get("candidate_id_set_sha256", ""))
    if h != mh or h != sh:
        raise RuntimeError(f"Candidate-ID hash mismatch: computed={h} manifest={mh} summary={sh}")
    if candidates["candidate_id"].duplicated().any():
        raise RuntimeError("Frozen candidate table contains duplicate candidate_id values")
    if len(upstream_outcomes) != len(candidates) or upstream_outcomes["candidate_id"].duplicated().any():
        raise RuntimeError("Frozen Phase-7I.1 candidate_outcomes is not one-to-one with candidates")
    return candidates.sort_values(["candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True), upstream_outcomes, summary, manifest, h


def rich_trade_aggregate(day_files, symbol: str):
    """Exact per-receive-second executed-trade primitives in bounded Arrow batches."""
    np, pd, _, pq = require_stack()
    cols = ["ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker"]
    parts = []
    for path in day_files.trades:
        pf = pq.ParquetFile(path)
        try:
            available = set(pf.schema_arrow.names)
            use = [c for c in cols if c in available]
            for batch in pf.iter_batches(batch_size=131072, columns=use):
                f = batch.to_pandas()
                if "symbol" in f.columns:
                    f = f[f["symbol"].astype(str) == str(symbol)].copy()
                if f.empty:
                    continue
                for c in ("ts_recv_ns", "price", "quantity"):
                    f[c] = pd.to_numeric(f[c], errors="coerce")
                f = f.dropna(subset=["ts_recv_ns", "price", "quantity"])
                f = f[(f["price"] > 0) & (f["quantity"] >= 0)].copy()
                if f.empty:
                    continue
                f["receive_second"] = (f["ts_recv_ns"].astype("int64") // 1_000_000_000).astype("int64")
                f["quote"] = f["price"] * f["quantity"]
                maker = f["buyer_is_maker"].fillna(False).astype(bool)
                f["signed"] = np.where(maker, -f["quote"], f["quote"])
                f["buy_quote"] = np.where(~maker, f["quote"], 0.0)
                f["sell_quote"] = np.where(maker, f["quote"], 0.0)
                f["buy_count"] = (~maker).astype("int64")
                f["sell_count"] = maker.astype("int64")
                f["quote_sq"] = f["quote"] * f["quote"]
                f["buy_trade_quote"] = np.where(~maker, f["quote"], 0.0)
                f["sell_trade_quote"] = np.where(maker, f["quote"], 0.0)
                g = f.groupby("receive_second", sort=True)
                part = g.agg(
                    trade_count=("quote", "size"),
                    trade_quote_volume=("quote", "sum"),
                    signed_aggressive_quote=("signed", "sum"),
                    buy_aggressive_quote=("buy_quote", "sum"),
                    sell_aggressive_quote=("sell_quote", "sum"),
                    buy_aggressive_count=("buy_count", "sum"),
                    sell_aggressive_count=("sell_count", "sum"),
                    trade_quote_sq_sum=("quote_sq", "sum"),
                    max_trade_quote=("quote", "max"),
                    max_buy_trade_quote=("buy_trade_quote", "max"),
                    max_sell_trade_quote=("sell_trade_quote", "max"),
                ).reset_index()
                parts.append(part)
        finally:
            close = getattr(pf, "close", None)
            if callable(close):
                close()
    if not parts:
        return pd.DataFrame(columns=[
            "receive_second", "trade_count", "trade_quote_volume", "signed_aggressive_quote",
            "buy_aggressive_quote", "sell_aggressive_quote", "buy_aggressive_count",
            "sell_aggressive_count", "trade_quote_sq_sum", "max_trade_quote",
            "max_buy_trade_quote", "max_sell_trade_quote",
        ])
    allp = pd.concat(parts, ignore_index=True)
    sums = allp.groupby("receive_second", sort=True).agg(
        trade_count=("trade_count", "sum"),
        trade_quote_volume=("trade_quote_volume", "sum"),
        signed_aggressive_quote=("signed_aggressive_quote", "sum"),
        buy_aggressive_quote=("buy_aggressive_quote", "sum"),
        sell_aggressive_quote=("sell_aggressive_quote", "sum"),
        buy_aggressive_count=("buy_aggressive_count", "sum"),
        sell_aggressive_count=("sell_aggressive_count", "sum"),
        trade_quote_sq_sum=("trade_quote_sq_sum", "sum"),
        max_trade_quote=("max_trade_quote", "max"),
        max_buy_trade_quote=("max_buy_trade_quote", "max"),
        max_sell_trade_quote=("max_sell_trade_quote", "max"),
    ).reset_index()
    return sums


class RichTradeContext:
    """Cumulative exact executed-trade context with concentration primitives."""
    ADDITIVE = (
        "trade_count", "trade_quote_volume", "signed_aggressive_quote",
        "buy_aggressive_quote", "sell_aggressive_quote", "buy_aggressive_count",
        "sell_aggressive_count", "trade_quote_sq_sum",
    )

    def __init__(self, frame):
        np, pd = require_numeric_stack()
        if frame is None or frame.empty:
            self.seconds = np.asarray([], dtype=np.int64)
            self.cums = {c: np.asarray([0.0]) for c in self.ADDITIVE}
            self.max_trade = np.asarray([], dtype=float)
            self.max_buy = np.asarray([], dtype=float)
            self.max_sell = np.asarray([], dtype=float)
            return
        f = frame.copy().sort_values("receive_second", kind="mergesort")
        self.seconds = pd.to_numeric(f["receive_second"], errors="coerce").fillna(-1).astype("int64").to_numpy()
        self.cums = {}
        for c in self.ADDITIVE:
            vals = pd.to_numeric(f.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
            self.cums[c] = np.concatenate([[0.0], np.cumsum(vals)])
        self.max_trade = pd.to_numeric(f.get("max_trade_quote", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        self.max_buy = pd.to_numeric(f.get("max_buy_trade_quote", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        self.max_sell = pd.to_numeric(f.get("max_sell_trade_quote", 0.0), errors="coerce").fillna(0.0).to_numpy(float)

    def _bounds(self, start_exclusive: int, end_inclusive: int) -> tuple[int, int]:
        lo = bisect.bisect_right(self.seconds, int(start_exclusive))
        hi = bisect.bisect_right(self.seconds, int(end_inclusive))
        return lo, hi

    def _sum(self, c: str, lo: int, hi: int) -> float:
        a = self.cums[c]
        return float(a[hi] - a[lo])

    @staticmethod
    def _div(a: float, b: float) -> float:
        return float(a / b) if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-12 else math.nan

    def range_summary(self, start_exclusive: int, end_inclusive: int, direction: int) -> dict[str, float]:
        np, _ = require_numeric_stack()
        lo, hi = self._bounds(start_exclusive, end_inclusive)
        total = self._sum("trade_quote_volume", lo, hi)
        signed = self._sum("signed_aggressive_quote", lo, hi)
        count = self._sum("trade_count", lo, hi)
        buy_q = self._sum("buy_aggressive_quote", lo, hi)
        sell_q = self._sum("sell_aggressive_quote", lo, hi)
        buy_n = self._sum("buy_aggressive_count", lo, hi)
        sell_n = self._sum("sell_aggressive_count", lo, hi)
        sq = self._sum("trade_quote_sq_sum", lo, hi)
        max_trade = float(np.max(self.max_trade[lo:hi])) if hi > lo else 0.0
        max_buy = float(np.max(self.max_buy[lo:hi])) if hi > lo else 0.0
        max_sell = float(np.max(self.max_sell[lo:hi])) if hi > lo else 0.0
        dnet = float(direction) * signed
        if int(direction) > 0:
            directional_q, counter_q = buy_q, sell_q
            directional_n, counter_n = buy_n, sell_n
            max_directional = max_buy
        else:
            directional_q, counter_q = sell_q, buy_q
            directional_n, counter_n = sell_n, buy_n
            max_directional = max_sell
        return {
            # Names required by the frozen 7E MechanicsTracker:
            "forward_trade_count": count,
            "forward_trade_quote_volume": total,
            "forward_signed_aggressive_quote": signed,
            "directional_net_aggressive_quote": dnet,
            "directional_aggressive_quote": directional_q,
            "counter_direction_aggressive_quote": counter_q,
            "forward_directional_aggressive_imbalance": self._div(dnet, total),
            # Rich executed-trade context:
            "directional_aggressive_trade_count": directional_n,
            "counter_direction_aggressive_trade_count": counter_n,
            "average_trade_quote": self._div(total, count),
            "directional_average_trade_quote": self._div(directional_q, directional_n),
            "max_trade_quote": max_trade,
            "max_directional_trade_quote": max_directional,
            "largest_trade_share": self._div(max_trade, total),
            "trade_size_hhi": self._div(sq, total * total),
            "directional_trade_count_share": self._div(directional_n, count),
        }


def enrich_window_row(row: dict[str, Any], trade_context: RichTradeContext, candidate_second: int, direction: int, horizon: int) -> dict[str, Any]:
    h = int(horizon)
    pre = trade_context.range_summary(int(candidate_second) - h, int(candidate_second), int(direction))
    for k, v in pre.items():
        row[f"pre_{k}"] = v
    elapsed = max(float(row.get("elapsed_sec") or h), 1e-12)
    pre_elapsed = float(h)
    fnet = safe_float(row.get("directional_net_aggressive_quote"))
    pnet = safe_float(pre.get("directional_net_aggressive_quote"))
    fvol = safe_float(row.get("forward_trade_quote_volume"))
    pvol = safe_float(pre.get("forward_trade_quote_volume"))
    fimb = safe_float(row.get("forward_directional_aggressive_imbalance"))
    pimb = safe_float(pre.get("forward_directional_aggressive_imbalance"))
    row["directional_net_flow_acceleration_quote_per_sec"] = (
        fnet / elapsed - pnet / pre_elapsed if math.isfinite(fnet) and math.isfinite(pnet) else math.nan
    )
    row["trade_quote_volume_acceleration_per_sec"] = (
        fvol / elapsed - pvol / pre_elapsed if math.isfinite(fvol) and math.isfinite(pvol) else math.nan
    )
    row["directional_imbalance_change"] = fimb - pimb if math.isfinite(fimb) and math.isfinite(pimb) else math.nan
    row["support_minus_opposing_retained_ratio"] = (
        safe_float(row.get("support__retained_initial_quote_ratio")) - safe_float(row.get("opposing__retained_initial_quote_ratio"))
        if math.isfinite(safe_float(row.get("support__retained_initial_quote_ratio"))) and math.isfinite(safe_float(row.get("opposing__retained_initial_quote_ratio")))
        else math.nan
    )
    row["opposing_minus_support_unrecovered_share"] = (
        safe_float(row.get("opposing__durable_unrecovered_loss_share_initial")) - safe_float(row.get("support__durable_unrecovered_loss_share_initial"))
        if math.isfinite(safe_float(row.get("opposing__durable_unrecovered_loss_share_initial"))) and math.isfinite(safe_float(row.get("support__durable_unrecovered_loss_share_initial")))
        else math.nan
    )
    daggr = safe_float(row.get("directional_aggressive_quote"))
    for prefix in ("opposing", "support"):
        refill = safe_float(row.get(f"{prefix}__same_price_refill_after_depletion_quote"))
        near = safe_float(row.get(f"{prefix}__noninitial_near_add_quote"))
        row[f"{prefix}__same_price_refill_to_directional_aggression_ratio"] = (
            refill / daggr if math.isfinite(refill) and math.isfinite(daggr) and daggr > 1e-12 else math.nan
        )
        row[f"{prefix}__near_restoration_to_directional_aggression_ratio"] = (
            (refill + near) / daggr if math.isfinite(refill) and math.isfinite(near) and math.isfinite(daggr) and daggr > 1e-12 else math.nan
        )
    return row


def build_day_files(phase6a, family_maps: Mapping[str, Mapping[date, tuple[Path, ...]]], days: Sequence[date]):
    def cat(family: str) -> tuple[Path, ...]:
        out: list[Path] = []
        for d in days:
            out.extend(family_maps[family].get(d, ()))
        return tuple(out)
    return phase6a.DayFiles(
        day=days[0],
        depth=cat("depth_updates"),
        snapshots=cat("book_snapshots"),
        trades=cat("trades"),
        liquidations=tuple(),
    )


def day_partition_paths(output_root: Path, candidate_date: str) -> tuple[Path, Path, Path, Path]:
    mech = output_root / "candidate_windows" / f"candidate_date={candidate_date}"
    lev = output_root / "level_audit" / f"candidate_date={candidate_date}"
    out = output_root / "decision_outcomes" / f"candidate_date={candidate_date}"
    return mech / "part.parquet", lev / "part.parquet", out / "part.parquet", mech / "manifest.json"


def execution_paths_for_days(phase7a, execution_root: Path, symbol: str, days: Sequence[date]) -> list[Path]:
    out = []
    for d in days:
        p = phase7a.execution_day_path(execution_root, symbol, d)
        if p.exists():
            out.append(p)
    return out


def day_input_signature(day_candidates, day_files, execution_files: Sequence[Path], candidate_date: str, candidate_hash: str) -> str:
    payload = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "candidate_date": candidate_date,
        "approved_candidate_hash": candidate_hash,
        "candidates": day_candidates.sort_values("candidate_id", kind="mergesort").to_dict("records"),
        "canonical_files": file_signature(day_files.depth + day_files.snapshots + day_files.trades),
        "execution_files": file_signature(execution_files),
        "fixed_horizons_sec": list(FIXED_HORIZONS_SEC),
        "outcome_horizons_sec": list(OUTCOME_HORIZONS_SEC),
        "frozen_near_bps": FROZEN_NEAR_BPS,
        "outer_band_bps": OUTER_BAND_BPS,
        "max_book_age_ms": MAX_BOOK_AGE_MS,
    }
    return sha256_json(payload)


def first_signed_passage_second(path_sec, directional_return_bps, threshold_bps: float, favorable: bool) -> int | None:
    thr = float(threshold_bps)
    for s, x in zip(path_sec, directional_return_bps):
        v = safe_float(x)
        if not math.isfinite(v):
            continue
        if (favorable and v + 1e-12 >= thr) or ((not favorable) and v - 1e-12 <= -thr):
            return int(s)
    return None


def build_decision_outcomes(day_candidates, phase7a, phase7i1, execution_root: Path, symbol: str, base: date):
    np, pd, _, _ = require_stack()
    days = [base]
    pnext = phase7a.execution_day_path(execution_root, symbol, base + timedelta(days=1))
    if pnext.exists():
        days.append(base + timedelta(days=1))
    frame = phase7a.load_execution_range(execution_root, symbol, days)
    sec, price, price_source = phase7a.build_catalog_price(frame, MAX_TRADE_STALE_SEC, "last")
    valid_end = phase7i1.contiguous_valid_end_indices(sec, price)
    rows = []
    for c in day_candidates.to_dict("records"):
        cs = int(c["candidate_second"])
        direction = int(c["direction"])
        origin = float(c["origin_price"])
        for h in FIXED_HORIZONS_SEC:
            ds = cs + int(h)
            di = int(np.searchsorted(sec, ds, side="left"))
            row: dict[str, Any] = {
                "builder_version": BUILDER_VERSION,
                "schema_version": SCHEMA_VERSION,
                "candidate_id": str(c["candidate_id"]),
                "episode_id": str(c["episode_id"]),
                "candidate_date": str(c["candidate_date"]),
                "direction": direction,
                "direction_label": str(c["direction_label"]),
                "observation_horizon_sec": int(h),
                "decision_second": ds,
                "decision_time": datetime.fromtimestamp(ds, tz=timezone.utc).replace(microsecond=0).isoformat(),
                "price_source": str(price_source),
            }
            if di >= len(sec) or int(sec[di]) != ds or not math.isfinite(float(price[di])) or float(price[di]) <= 0:
                row["decision_price_valid"] = False
                row["decision_invalid_reason"] = "DECISION_PRICE_MISSING_OR_STALE"
                rows.append(row)
                continue
            dp = float(price[di])
            disp = direction * (dp / origin - 1.0) * 10_000.0
            row.update({
                "decision_price_valid": True,
                "decision_invalid_reason": None,
                "decision_price": dp,
                "decision_displacement_from_origin_bps": float(disp),
                "extra_displacement_since_35bp_candidate_bps": float(disp - float(c["candidate_displacement_bps"])),
                "nominal_remaining_to_50_bps": float(50.0 - disp),
                "nominal_remaining_to_75_bps": float(75.0 - disp),
                "nominal_remaining_to_100_bps": float(100.0 - disp),
            })
            seg_end = int(valid_end[di]) if di < len(valid_end) else -1
            for oh in OUTCOME_HORIZONS_SEC:
                target_s = ds + int(oh)
                hi = int(np.searchsorted(sec, target_s, side="right") - 1)
                fully = hi >= di and hi < len(sec) and int(sec[hi]) == target_s and seg_end >= hi
                use_hi = min(seg_end, hi) if seg_end >= di and hi >= di else di
                psec = sec[di : use_hi + 1]
                ppx = price[di : use_hi + 1]
                dr_dec = direction * (ppx / dp - 1.0) * 10_000.0
                dr_origin = direction * (ppx / origin - 1.0) * 10_000.0
                row[f"h{oh}_fully_observed"] = bool(fully)
                row[f"h{oh}_observed_seconds"] = int(psec[-1] - ds) if len(psec) else 0
                row[f"h{oh}_mfe_from_decision_bps"] = float(np.nanmax(dr_dec)) if len(dr_dec) else math.nan
                row[f"h{oh}_mae_from_decision_bps"] = float(max(0.0, -float(np.nanmin(dr_dec)))) if len(dr_dec) else math.nan
                for t in (50.0, 75.0, 100.0):
                    hit = first_signed_passage_second(psec, dr_origin, t, favorable=True)
                    name = int(t)
                    row[f"h{oh}_origin_hit_{name}_second"] = hit
                    row[f"h{oh}_origin_reach_{name}"] = True if hit is not None else (False if fully else None)
                    row[f"h{oh}_seconds_to_origin_{name}_from_decision"] = int(hit - ds) if hit is not None else math.nan
                for t in FAVORABLE_PASSAGES_BPS:
                    hit = first_signed_passage_second(psec, dr_dec, t, favorable=True)
                    name = int(t)
                    row[f"h{oh}_decision_favorable_{name}_hit_second"] = hit
                    row[f"h{oh}_decision_favorable_{name}_reached"] = True if hit is not None else (False if fully else None)
                    row[f"h{oh}_seconds_to_favorable_{name}"] = int(hit - ds) if hit is not None else math.nan
                for t in ADVERSE_PASSAGES_BPS:
                    hit = first_signed_passage_second(psec, dr_dec, t, favorable=False)
                    name = int(t)
                    row[f"h{oh}_decision_adverse_{name}_hit_second"] = hit
                    row[f"h{oh}_decision_adverse_{name}_reached"] = True if hit is not None else (False if fully else None)
                    row[f"h{oh}_seconds_to_adverse_{name}"] = int(hit - ds) if hit is not None else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def process_candidate_date(candidate_date: str, day_candidates, candidate_hash: str, phase6a, phase7a, phase7e, phase7i1, family_maps, args):
    np, pd, _, pq = require_stack()
    base = date.fromisoformat(candidate_date)
    max_candidate = int(pd.to_numeric(day_candidates["candidate_second"], errors="raise").max())
    max_end = max_candidate + max(FIXED_HORIZONS_SEC)
    end_date = datetime.fromtimestamp(max_end, tz=timezone.utc).date()
    scan_days = [base]
    if end_date > base:
        scan_days.append(base + timedelta(days=1))
    day_files = build_day_files(phase6a, family_maps, scan_days)
    if not day_files.depth or not day_files.snapshots:
        raise RuntimeError(f"Missing canonical depth/snapshot files for candidate date {candidate_date}")
    exec_days = [base]
    if phase7a.execution_day_path(args.execution_root, args.symbol, base + timedelta(days=1)).exists():
        exec_days.append(base + timedelta(days=1))
    exec_files = execution_paths_for_days(phase7a, args.execution_root, args.symbol, exec_days)
    if not exec_files:
        raise RuntimeError(f"Missing Phase-6A execution-price partitions for {candidate_date}")

    mech_path, level_path, outcome_path, manifest_path = day_partition_paths(args.output_root, candidate_date)
    sig = day_input_signature(day_candidates, day_files, exec_files, candidate_date, candidate_hash)
    if not args.force_day and mech_path.exists() and level_path.exists() and outcome_path.exists() and manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            m = {}
        if str(m.get("builder_version")) == BUILDER_VERSION and str(m.get("input_signature")) == sig:
            diag = {k: m.get(k) for k in (
                "trade_aggregation_seconds", "raw_replay_seconds", "outcome_seconds",
                "depth_rows_decoded", "depth_levels_decoded", "depth_arrow_batches",
                "tracked_depth_messages", "tracked_exact_changes",
            )}
            return pq.read_table(mech_path).to_pandas(), pq.read_table(level_path).to_pandas(), pq.read_table(outcome_path).to_pandas(), True, diag

    trade_t0 = time.perf_counter()
    rich_trades = rich_trade_aggregate(day_files, args.symbol)
    trade_context = RichTradeContext(rich_trades)
    trade_seconds = time.perf_counter() - trade_t0

    # Candidate rows are adapted to the frozen 7E MechanicsTracker interface.
    anchors = []
    for c in day_candidates.to_dict("records"):
        cs = int(c["candidate_second"])
        ev = {
            "event_id": str(c["candidate_id"]),
            "onset_date": str(c["candidate_date"]),
            "direction": int(c["direction"]),
            "continuation_stratum": "ALL_MARKET_CAUSAL_35BP_CANDIDATE",
            "candidate_id": str(c["candidate_id"]),
        }
        anchors.append(((cs + 1) * 1_000_000_000 - 1, cs, ev, c))
    anchors.sort(key=lambda x: (x[0], x[2]["event_id"]))

    price_scale = float(args.price_scale)
    book = phase6a.LocalBook(price_scale=price_scale)
    fast_stats = phase6a.FastDepthStats()
    gap_records = []
    active = []
    profile_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    anchor_idx = 0
    stop_ns = (max_end + 1) * 1_000_000_000 - 1
    tracked_depth_messages = 0
    tracked_exact_changes = 0
    replay_t0 = time.perf_counter()

    def instantiate_before(ts_ns: int) -> None:
        nonlocal anchor_idx
        while anchor_idx < len(anchors) and anchors[anchor_idx][0] < ts_ns:
            _ans, cs, ev, c = anchors[anchor_idx]
            # make_tracker recognizes unknown stage names and therefore emits
            # only fixed 1/3/5/10-second endpoints, exactly what 7I.4 requires.
            tr, invalid = phase7e.make_tracker(
                ev, "CAUSAL_35BP", cs, book, trade_context,
                args.max_book_age_ms, price_scale,
            )
            for r in invalid:
                r["candidate_id"] = str(c["candidate_id"])
                r["candidate_date"] = str(c["candidate_date"])
                r["observation_horizon_sec"] = int(str(r["window_label"]).split("_")[1][:-1]) if str(r.get("window_label", "")).startswith("FWD_") else math.nan
                h = int(r["observation_horizon_sec"])
                r["decision_second"] = int(cs) + h
                r["episode_id"] = str(c["episode_id"])
                r["candidate_price"] = float(c["candidate_price"])
                r["origin_price"] = float(c["origin_price"])
                r["candidate_displacement_bps"] = float(c["candidate_displacement_bps"])
                r["seconds_release_to_candidate"] = int(c["seconds_release_to_candidate"])
                r.update(trade_context.range_summary(int(cs), int(cs) + h, int(c["direction"])))
                enrich_window_row(r, trade_context, int(cs), int(c["direction"]), h)
                r["builder_version"] = BUILDER_VERSION
                r["schema_version"] = SCHEMA_VERSION
                profile_rows.append(r)
            if tr is not None:
                # Attach candidate metadata without changing frozen tracker math.
                tr._phase7i4_candidate = c
                active.append(tr)
            anchor_idx += 1

    def finish_done() -> None:
        nonlocal active
        remain = []
        for tr in active:
            if tr.done:
                c = tr._phase7i4_candidate
                for r in tr.rows:
                    label = str(r.get("window_label", ""))
                    h = int(label.split("_")[1][:-1]) if label.startswith("FWD_") else int(round(safe_float(r.get("elapsed_sec"))))
                    r["candidate_id"] = str(c["candidate_id"])
                    r["candidate_date"] = str(c["candidate_date"])
                    r["episode_id"] = str(c["episode_id"])
                    r["observation_horizon_sec"] = h
                    r["decision_second"] = int(c["candidate_second"]) + h
                    r["candidate_price"] = float(c["candidate_price"])
                    r["origin_price"] = float(c["origin_price"])
                    r["candidate_displacement_bps"] = float(c["candidate_displacement_bps"])
                    r["seconds_release_to_candidate"] = int(c["seconds_release_to_candidate"])
                    r["builder_version"] = BUILDER_VERSION
                    r["schema_version"] = SCHEMA_VERSION
                    enrich_window_row(r, trade_context, int(c["candidate_second"]), int(c["direction"]), h)
                    profile_rows.append(r)
                for lr in tr.level_audit_rows(book):
                    lr["candidate_id"] = str(c["candidate_id"])
                    lr["candidate_date"] = str(c["candidate_date"])
                    lr["builder_version"] = BUILDER_VERSION
                    lr["schema_version"] = SCHEMA_VERSION
                    level_rows.append(lr)
            else:
                remain.append(tr)
        active = remain

    for rec in phase6a.unified_book_events(day_files, args.symbol, price_scale, fast_stats):
        ts_ns = int(rec["ts_recv_ns"])
        if ts_ns > stop_ns:
            instantiate_before(ts_ns)
            for tr in active:
                tr.finalize_due_before(ts_ns, book, trade_context)
            finish_done()
            break

        instantiate_before(ts_ns)
        for tr in active:
            tr.finalize_due_before(ts_ns, book, trade_context)

        kind = rec.get("_kind")
        if kind == "SNAPSHOT":
            book.apply_snapshot(rec)
        elif kind == "DEPTH":
            applicable = phase7e.depth_row_is_applicable(book, rec)
            track_changes = bool(active) and any(
                tr.censored_ns is None and not tr.done and ts_ns > tr.anchor_end_ns for tr in active
            )
            changes = phase7e.extract_row_changes(book, rec) if applicable and track_changes else []
            if applicable and track_changes:
                tracked_depth_messages += 1
                tracked_exact_changes += len(changes)
            was_valid = bool(book.valid)
            book.apply_depth(rec, gap_records, candidate_date)
            still_valid = bool(book.valid)
            if applicable and was_valid and still_valid:
                for tr in active:
                    tr.on_changes(ts_ns, changes)
            elif was_valid and not still_valid:
                reason = gap_records[-1].reason if gap_records else "BOOK_INVALIDATED"
                for tr in active:
                    if ts_ns > tr.anchor_end_ns and not tr.done:
                        tr.censor(ts_ns, reason)

        for tr in active:
            tr.finalize_due_through(ts_ns, book, trade_context)
        finish_done()

    instantiate_before(stop_ns + 1)
    for tr in active:
        tr.finalize_due_through(stop_ns, book, trade_context)
        if not tr.done:
            tr.finalize_all(book, trade_context)
    finish_done()
    replay_seconds = time.perf_counter() - replay_t0

    mechanics = pd.DataFrame(profile_rows)
    levels = pd.DataFrame(level_rows)
    # Ensure every candidate has every horizon row, even if anchor book invalid.
    expected = {(str(c), int(h)) for c in day_candidates["candidate_id"].astype(str) for h in FIXED_HORIZONS_SEC}
    observed = set()
    if not mechanics.empty:
        observed = set(zip(mechanics["candidate_id"].astype(str), pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce").fillna(-1).astype(int)))
    missing = sorted(expected - observed)
    if missing:
        cmap = {str(r["candidate_id"]): r for r in day_candidates.to_dict("records")}
        extra = []
        for cid, h in missing:
            c = cmap[cid]
            rr = {
                "builder_version": BUILDER_VERSION,
                "schema_version": SCHEMA_VERSION,
                "candidate_id": cid,
                "event_id": cid,
                "episode_id": str(c["episode_id"]),
                "candidate_date": str(c["candidate_date"]),
                "direction": int(c["direction"]),
                "direction_label": str(c["direction_label"]),
                "stage": "CAUSAL_35BP",
                "stage_second": int(c["candidate_second"]),
                "window_label": f"FWD_{h}S",
                "window_type": "FIXED_HORIZON",
                "window_end_second": int(c["candidate_second"]) + h,
                "elapsed_sec": float(h),
                "observation_horizon_sec": h,
                "decision_second": int(c["candidate_second"]) + h,
                "candidate_price": float(c["candidate_price"]),
                "origin_price": float(c["origin_price"]),
                "candidate_displacement_bps": float(c["candidate_displacement_bps"]),
                "seconds_release_to_candidate": int(c["seconds_release_to_candidate"]),
                "window_valid": False,
                "invalid_reason": "TRACKER_ROW_MISSING",
            }
            rr.update(trade_context.range_summary(int(c["candidate_second"]), int(c["candidate_second"]) + h, int(c["direction"])))
            enrich_window_row(rr, trade_context, int(c["candidate_second"]), int(c["direction"]), h)
            extra.append(rr)
        mechanics = pd.concat([mechanics, pd.DataFrame(extra)], ignore_index=True) if not mechanics.empty else pd.DataFrame(extra)

    outcome_t0 = time.perf_counter()
    outcomes = build_decision_outcomes(day_candidates, phase7a, phase7i1, args.execution_root, args.symbol, base)
    outcome_seconds = time.perf_counter() - outcome_t0

    # Decision price/overshoot is causal by the mechanics row's own observation
    # timestamp.  Copy only these causal fields into the causal mechanics table;
    # future labels remain physically isolated in outcomes.
    causal_decision_cols = [
        "candidate_id", "observation_horizon_sec", "decision_price_valid",
        "decision_invalid_reason", "decision_price",
        "decision_displacement_from_origin_bps",
        "extra_displacement_since_35bp_candidate_bps",
        "nominal_remaining_to_50_bps", "nominal_remaining_to_75_bps",
        "nominal_remaining_to_100_bps",
    ]
    if not mechanics.empty and not outcomes.empty:
        present = [c for c in causal_decision_cols if c in outcomes.columns]
        mechanics = mechanics.merge(
            outcomes[present],
            on=["candidate_id", "observation_horizon_sec"],
            how="left", validate="one_to_one",
        )

    if not mechanics.empty:
        mechanics["day_input_signature"] = sig
        mechanics = mechanics.sort_values(["candidate_id", "observation_horizon_sec"], kind="mergesort").reset_index(drop=True)
    if not levels.empty:
        levels["day_input_signature"] = sig
    if not outcomes.empty:
        outcomes["day_input_signature"] = sig
        outcomes = outcomes.sort_values(["candidate_id", "observation_horizon_sec"], kind="mergesort").reset_index(drop=True)

    write_parquet_atomic(mechanics, mech_path)
    write_parquet_atomic(levels, level_path)
    write_parquet_atomic(outcomes, outcome_path)
    atomic_write_json(manifest_path, {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "candidate_date": candidate_date,
        "input_signature": sig,
        "candidate_count": int(day_candidates["candidate_id"].nunique()),
        "mechanics_rows": int(len(mechanics)),
        "valid_mechanics_rows": int(mechanics["window_valid"].fillna(False).sum()) if len(mechanics) else 0,
        "level_rows": int(len(levels)),
        "decision_outcome_rows": int(len(outcomes)),
        "gap_records_observed_during_scan": int(len(gap_records)),
        "trade_aggregation_seconds": float(trade_seconds),
        "raw_replay_seconds": float(replay_seconds),
        "outcome_seconds": float(outcome_seconds),
        "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0)),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0)),
        "depth_arrow_batches": int(getattr(fast_stats, "arrow_batches", 0)),
        "tracked_depth_messages": int(tracked_depth_messages),
        "tracked_exact_changes": int(tracked_exact_changes),
        "generated_at": utc_now_iso(),
    })
    diag = {
        "trade_aggregation_seconds": trade_seconds,
        "raw_replay_seconds": replay_seconds,
        "outcome_seconds": outcome_seconds,
        "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0)),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0)),
        "depth_arrow_batches": int(getattr(fast_stats, "arrow_batches", 0)),
        "tracked_depth_messages": tracked_depth_messages,
        "tracked_exact_changes": tracked_exact_changes,
    }
    return mechanics, levels, outcomes, False, diag


def build_coverage(mechanics, candidates):
    _, pd, _, _ = require_stack()
    rows = []
    for h in FIXED_HORIZONS_SEC:
        m = mechanics[pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == h].copy()
        valid = m["window_valid"].fillna(False).astype(bool) if len(m) else pd.Series([], dtype=bool)
        rows.append({
            "observation_horizon_sec": h,
            "candidate_count": int(len(candidates)),
            "row_count": int(len(m)),
            "valid_book_mechanics": int(valid.sum()),
            "book_coverage_rate": float(valid.mean()) if len(valid) else math.nan,
            "invalid_book_mechanics": int((~valid).sum()) if len(valid) else 0,
        })
    return pd.DataFrame(rows)


def build_day_quality(mechanics, candidates):
    _, pd, _, _ = require_stack()
    rows = []
    for d, c in candidates.groupby("candidate_date", sort=True):
        m = mechanics[(mechanics["candidate_date"].astype(str) == str(d)) & (pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC)]
        rows.append({
            "candidate_date": str(d),
            "candidates": int(len(c)),
            "up": int((pd.to_numeric(c["direction"], errors="coerce") > 0).sum()),
            "down": int((pd.to_numeric(c["direction"], errors="coerce") < 0).sum()),
            "primary_rows": int(len(m)),
            "primary_valid": int(m["window_valid"].fillna(False).sum()) if len(m) else 0,
            "primary_coverage_rate": float(m["window_valid"].fillna(False).mean()) if len(m) else math.nan,
        })
    return pd.DataFrame(rows)


def build_schema_report(mechanics, outcomes):
    _, pd, _, _ = require_stack()
    rows = []
    causal_prefixes = ("h300_", "h600_", "h900_", "h1800_", "episode_reach_", "mfe_", "mae_")
    for dataset, df in (("CAUSAL_MECHANICS", mechanics), ("FUTURE_OUTCOMES", outcomes)):
        for c in df.columns:
            rows.append({
                "dataset": dataset,
                "column": str(c),
                "role": "FUTURE_LABEL" if dataset == "FUTURE_OUTCOMES" else "CAUSAL_BY_DECISION_TIME",
                "looks_like_future_label": bool(str(c).startswith(causal_prefixes) or "_reach_" in str(c) or "mfe" in str(c).lower() or "mae" in str(c).lower()),
            })
    return pd.DataFrame(rows)


def check_invariants(candidates, mechanics, outcomes, approved_hash: str, level_summary: Mapping[str, Any], partial_scope: bool):
    _, pd, _, _ = require_stack()
    rows = []
    def add(name: str, ok: bool, observed: Any, expected: Any):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    ids = set(candidates["candidate_id"].astype(str))
    mids = set(mechanics["candidate_id"].astype(str)) if len(mechanics) else set()
    oids = set(outcomes["candidate_id"].astype(str)) if len(outcomes) else set()
    add("candidate_ids_unique", not candidates["candidate_id"].duplicated().any(), int(candidates["candidate_id"].duplicated().sum()), 0)
    add("mechanics_keys_unique", not mechanics.duplicated(["candidate_id", "observation_horizon_sec"]).any() if len(mechanics) else False, int(mechanics.duplicated(["candidate_id", "observation_horizon_sec"]).sum()) if len(mechanics) else -1, 0)
    add("decision_outcome_keys_unique", not outcomes.duplicated(["candidate_id", "observation_horizon_sec"]).any() if len(outcomes) else False, int(outcomes.duplicated(["candidate_id", "observation_horizon_sec"]).sum()) if len(outcomes) else -1, 0)
    add("all_processed_candidates_preserved_in_mechanics", mids == ids, f"mechanics={len(mids)} candidates={len(ids)}", "same ID set")
    add("all_processed_candidates_preserved_in_outcomes", oids == ids, f"outcomes={len(oids)} candidates={len(ids)}", "same ID set")
    add("four_horizon_rows_per_candidate", len(mechanics) == len(candidates) * len(FIXED_HORIZONS_SEC), len(mechanics), len(candidates) * len(FIXED_HORIZONS_SEC))
    add("four_decision_outcome_rows_per_candidate", len(outcomes) == len(candidates) * len(FIXED_HORIZONS_SEC), len(outcomes), len(candidates) * len(FIXED_HORIZONS_SEC))
    decision_ok = True
    if len(mechanics):
        cs = mechanics["candidate_id"].map(candidates.set_index("candidate_id")["candidate_second"])
        h = pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce")
        ds = pd.to_numeric(mechanics["decision_second"], errors="coerce")
        decision_ok = bool(((ds - pd.to_numeric(cs, errors="coerce")) == h).fillna(False).all())
    add("mechanics_decision_time_is_candidate_plus_horizon", decision_ok, int(decision_ok), 1)
    futureish = [c for c in mechanics.columns if c.startswith(("h300_", "h600_", "h900_", "h1800_", "episode_reach_")) or "mfe_from" in c.lower() or "mae_from" in c.lower()]
    add("future_labels_absent_from_causal_mechanics", len(futureish) == 0, futureish[:20], [])
    add("frozen_level_initial_qty_positive", int(level_summary.get("nonpositive_initial_qty", 0)) == 0, int(level_summary.get("nonpositive_initial_qty", 0)), 0)
    roles = sorted(level_summary.get("roles", set()))
    add("frozen_level_roles_known", set(roles).issubset({"SUPPORT", "OPPOSING"}) and bool(roles), roles, "SUPPORT/OPPOSING")
    primary = mechanics[pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]
    cov = float(primary["window_valid"].fillna(False).mean()) if len(primary) else 0.0
    if partial_scope:
        add("primary_3s_book_coverage_smoke", bool(len(primary)) and int(primary["window_valid"].fillna(False).sum()) > 0, f"{cov:.4f}", ">0 valid primary windows; full-run >=95% gate deferred")
    else:
        add("primary_3s_book_coverage", cov >= MIN_PRIMARY_BOOK_COVERAGE, f"{cov:.4f}", f">={MIN_PRIMARY_BOOK_COVERAGE:.2f}")
    if not partial_scope:
        add("approved_full_candidate_count", len(candidates) == EXPECTED_FULL_CANDIDATES, len(candidates), EXPECTED_FULL_CANDIDATES)
        add("approved_full_up_down", (int((candidates["direction"] > 0).sum()), int((candidates["direction"] < 0).sum())) == (EXPECTED_FULL_UP, EXPECTED_FULL_DOWN), (int((candidates["direction"] > 0).sum()), int((candidates["direction"] < 0).sum())), (EXPECTED_FULL_UP, EXPECTED_FULL_DOWN))
        add("approved_candidate_id_hash", candidate_id_hash(candidates) == approved_hash, candidate_id_hash(candidates), approved_hash)
    return pd.DataFrame(rows)


def build_findings(summary: Mapping[str, Any], coverage) -> str:
    primary = coverage[coverage["observation_horizon_sec"] == PRIMARY_HORIZON_SEC]
    pcov = safe_float(primary.iloc[0]["book_coverage_rate"]) if len(primary) else math.nan
    return f"""# Phase 7I.4 — Causal Candidate Participant-Interaction Materialization\n\n## Status\n\n- Builder: `{BUILDER_VERSION}`\n- Status: **{summary['status']}**\n- Scope: `{summary['run_scope']}`\n- Candidates processed: **{summary['candidate_count']:,}**\n- Candidate dates processed: **{summary['candidate_dates']}**\n- Causal mechanics rows: **{summary['mechanics_rows']:,}**\n- Decision-outcome rows: **{summary['decision_outcome_rows']:,}**\n- Primary 3-second book coverage: **{pcov:.2%}**\n\n## Scientific role\n\nThis phase materializes the frozen all-market causal perps candidate substrate. It does **not** select a mechanism threshold or claim alpha. Exact book response and executed-trade pressure are measured together at 1/3/5/10-second live-observable decision times. Future continuation/MFE/MAE labels are stored in a physically separate dataset.\n\n## Frozen candidate contract\n\nThe full approved Phase-7I.1 population is 2,729 causal first-35-bp candidates from the predeclared 10-bp directional-change episode ontology. Candidate IDs are not regenerated or filtered here. The approved ID-set SHA-256 is carried into metadata for all downstream analyses.\n\n## Next cheap analyses\n\nUse the frozen causal mechanics and separate outcomes to evaluate transparent conditional continuation probabilities, day/chronological robustness, MFE/MAE, first favorable/adverse passage, and economic reward:risk. Do not replay raw depth for those questions.\n"""


def self_test() -> None:
    np, pd = require_numeric_stack()
    f = pd.DataFrame({
        "receive_second": [101, 102, 103],
        "trade_count": [2, 1, 1],
        "trade_quote_volume": [300.0, 200.0, 100.0],
        "signed_aggressive_quote": [100.0, -200.0, 100.0],
        "buy_aggressive_quote": [200.0, 0.0, 100.0],
        "sell_aggressive_quote": [100.0, 200.0, 0.0],
        "buy_aggressive_count": [1, 0, 1],
        "sell_aggressive_count": [1, 1, 0],
        "trade_quote_sq_sum": [50_000.0, 40_000.0, 10_000.0],
        "max_trade_quote": [200.0, 200.0, 100.0],
        "max_buy_trade_quote": [200.0, 0.0, 100.0],
        "max_sell_trade_quote": [100.0, 200.0, 0.0],
    })
    ctx = RichTradeContext(f)
    up = ctx.range_summary(100, 103, 1)
    assert abs(up["forward_trade_quote_volume"] - 600.0) < 1e-9
    assert abs(up["directional_aggressive_quote"] - 300.0) < 1e-9
    assert abs(up["counter_direction_aggressive_quote"] - 300.0) < 1e-9
    assert abs(up["largest_trade_share"] - (200.0 / 600.0)) < 1e-9
    down = ctx.range_summary(100, 103, -1)
    assert abs(down["directional_aggressive_quote"] - 300.0) < 1e-9
    sec = np.asarray([10, 11, 12, 13], dtype=np.int64)
    dr = np.asarray([0.0, 5.0, 12.0, -11.0])
    assert first_signed_passage_second(sec, dr, 10.0, True) == 12
    assert first_signed_passage_second(sec, dr, 10.0, False) == 13
    print("SELF-TEST PASS")
    print("Validated executed-trade direction/concentration arithmetic, causal decision-window aggregation, and favorable/adverse first-passage helpers.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Build Phase-7I.4 reusable causal candidate participant-interaction dataset")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--phase7e-script", type=Path, default=DEFAULT_PHASE7E_SCRIPT)
    p.add_argument("--phase7i1-script", type=Path, default=DEFAULT_PHASE7I1_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-scale", type=float, default=PRICE_SCALE)
    p.add_argument("--max-book-age-ms", type=int, default=MAX_BOOK_AGE_MS)
    p.add_argument("--force-day", action="store_true", help="Rebuild per-candidate-date checkpoints even when signatures match")
    p.add_argument("--max-candidate-dates", type=int, default=None, help="Smoke-test only: process the first N candidate dates")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.price_scale <= 0 or args.max_book_age_ms <= 0:
        raise SystemExit("--price-scale and --max-book-age-ms must be positive")
    if abs(args.price_scale - PRICE_SCALE) > 1e-12:
        raise SystemExit(f"Phase 7I.4 is frozen to price_scale={PRICE_SCALE}; received {args.price_scale}")
    if args.max_book_age_ms != MAX_BOOK_AGE_MS:
        raise SystemExit(f"Phase 7I.4 is frozen to max_book_age_ms={MAX_BOOK_AGE_MS}; received {args.max_book_age_ms}")

    np, pd, pa, pq = require_stack()
    phase6a, phase7a, phase7e, phase7i1 = load_modules(args)
    candidates_all, upstream_outcomes, candidate_summary, candidate_manifest, approved_hash = load_frozen_candidates(args)

    print("Building Phase 7I.4 causal candidate participant-interaction substrate...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-6A      : {phase6a.BUILDER_VERSION}")
    print(f"Frozen Phase-7A      : {phase7a.BUILDER_VERSION}")
    print(f"Frozen Phase-7E      : {phase7e.BUILDER_VERSION}")
    print(f"Frozen Phase-7I.1    : {phase7i1.BUILDER_VERSION}")
    print(f"Approved candidates  : {len(candidates_all):,} ({EXPECTED_FULL_UP:,} UP / {EXPECTED_FULL_DOWN:,} DOWN)")
    print(f"Candidate ID hash    : {approved_hash[:16]}...")
    print(f"Decision horizons    : {','.join(str(x) for x in FIXED_HORIZONS_SEC)}s (primary={PRIMARY_HORIZON_SEC}s)")
    print("Research role        : all-market causal perps measurement substrate; not prospective alpha")

    print("[1/8] Freezing approved candidate contract and separating future labels...")
    all_dates = sorted(candidates_all["candidate_date"].astype(str).unique())
    selected_dates = list(all_dates)
    if args.max_candidate_dates is not None:
        selected_dates = selected_dates[: max(0, int(args.max_candidate_dates))]
    partial_scope = len(selected_dates) < len(all_dates)
    candidates = candidates_all[candidates_all["candidate_date"].astype(str).isin(selected_dates)].copy().reset_index(drop=True)
    print(f"      candidate dates={len(selected_dates)}" + (f" of {len(all_dates)} (PARTIAL SMOKE SCOPE)" if partial_scope else ""))
    print(f"      candidates in scope={len(candidates):,}")

    print("[2/8] Discovering canonical depth/snapshot/trade partitions...")
    family_maps = {fam: phase6a.discover_family_days(args.canonical_root, fam) for fam in ("depth_updates", "book_snapshots", "trades")}

    print("[3/8] Replaying authoritative book once per candidate date and materializing participant interactions...")
    mechanics_parts = []
    outcome_parts = []
    level_summary = {"row_count": 0, "nonpositive_initial_qty": 0, "roles": set()}
    reused_days = 0
    for i, d in enumerate(selected_dates, 1):
        dc = candidates[candidates["candidate_date"].astype(str) == str(d)].copy()
        mech_day, level_day, out_day, reused, diag = process_candidate_date(
            str(d), dc, approved_hash, phase6a, phase7a, phase7e, phase7i1, family_maps, args
        )
        mechanics_parts.append(mech_day)
        outcome_parts.append(out_day)
        if not level_day.empty:
            level_summary["row_count"] += int(len(level_day))
            iq = pd.to_numeric(level_day["initial_qty"], errors="coerce")
            level_summary["nonpositive_initial_qty"] += int((iq <= 0).sum())
            level_summary["roles"].update(str(x) for x in level_day["side_role"].dropna().unique())
        reused_days += int(reused)
        valid = int(mech_day["window_valid"].fillna(False).sum()) if len(mech_day) else 0
        line = f"      [{i:02d}/{len(selected_dates):02d}] {d}: candidates={len(dc):4d} windows={len(mech_day):5d} valid={valid:5d} {'REUSED' if reused else 'BUILT'}"
        if not reused:
            line += (
                f" | trades={safe_float(diag.get('trade_aggregation_seconds')):.1f}s"
                f" replay={safe_float(diag.get('raw_replay_seconds')):.1f}s"
                f" outcomes={safe_float(diag.get('outcome_seconds')):.1f}s"
                f" depth_rows={int(diag.get('depth_rows_decoded') or 0):,}"
                f" tracked_msgs={int(diag.get('tracked_depth_messages') or 0):,}"
            )
        print(line)
        del level_day, dc
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

    mechanics = pd.concat(mechanics_parts, ignore_index=True) if mechanics_parts else pd.DataFrame()
    decision_outcomes = pd.concat(outcome_parts, ignore_index=True) if outcome_parts else pd.DataFrame()

    print("[4/8] Building reusable causal candidate dataset and data-quality reports...")
    # Self-contained causal candidate rows in the downstream substrate.
    candidate_inputs = candidates.copy()
    coverage = build_coverage(mechanics, candidates)
    day_quality = build_day_quality(mechanics, candidates)
    schema_report = build_schema_report(mechanics, decision_outcomes)

    print("[5/8] Checking substrate invariants...")
    checks = check_invariants(candidates, mechanics, decision_outcomes, approved_hash, level_summary, partial_scope)
    failed = checks[checks["status"] == "FAIL"]
    for r in checks.to_dict("records"):
        print(f"      {r['status']:<4} {r['check']}: observed={r['observed']} expected={r['expected']}")

    print("[6/8] Writing frozen datasets and approved manifest...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(candidate_inputs, args.output_root / "candidate_inputs.parquet")
    write_parquet_atomic(mechanics, args.output_root / "candidate_mechanics.parquet")
    write_parquet_atomic(decision_outcomes, args.output_root / "candidate_decision_outcomes.parquet")
    # Preserve original 7I.1 candidate-level future labels as a separate audit convenience.
    upstream_in_scope = upstream_outcomes[upstream_outcomes["candidate_id"].astype(str).isin(set(candidates["candidate_id"].astype(str)))].copy()
    write_parquet_atomic(upstream_in_scope, args.output_root / "candidate_7i1_outcomes.parquet")

    status = ("SMOKE_PASS" if partial_scope else "PASS") if failed.empty else "FAIL"
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "run_scope": "PARTIAL_SMOKE" if partial_scope else "FULL",
        "symbol": args.symbol,
        "research_role": "ALL_MARKET_CAUSAL_PERPS_PARTICIPANT_INTERACTION_SUBSTRATE_NOT_ALPHA",
        "candidate_count": int(len(candidates)),
        "candidate_dates": int(len(selected_dates)),
        "total_approved_candidates": int(len(candidates_all)),
        "approved_candidate_id_set_sha256": approved_hash,
        "mechanics_rows": int(len(mechanics)),
        "valid_mechanics_rows": int(mechanics["window_valid"].fillna(False).sum()) if len(mechanics) else 0,
        "decision_outcome_rows": int(len(decision_outcomes)),
        "level_audit_rows": int(level_summary["row_count"]),
        "reused_candidate_date_checkpoints": int(reused_days),
        "failed_invariants": int(len(failed)),
        "generated_at": utc_now_iso(),
    }
    approved_manifest = {
        "schema_name": SCHEMA_NAME,
        "builder_version": BUILDER_VERSION,
        "upstream_candidate_builder": REQUIRED_PHASE7I1_VERSION,
        "candidate_count": int(len(candidates_all)),
        "candidate_id_set_sha256": approved_hash,
        "candidate_freeze_status": "SCIENTIFICALLY_APPROVED_FOR_7I_PARTICIPANT_INTERACTION",
        "approval_basis": "Phase-7I.1/7I.2 audit reviewed: manageable causal population, balanced directions, no duplicate explosion, and ~97.3%/98.7% coverage of retrospective 75+/100+ physical moves. Candidate ontology is not retuned in Phase 7I.4.",
        "upstream_manifest_freeze_status": candidate_manifest.get("freeze_status"),
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", {
        **summary,
        "canonical_root": str(args.canonical_root.resolve()),
        "execution_root": str(args.execution_root.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "fixed_horizons_sec": list(FIXED_HORIZONS_SEC),
        "primary_horizon_sec": PRIMARY_HORIZON_SEC,
        "outcome_horizons_sec": list(OUTCOME_HORIZONS_SEC),
        "favorable_passages_bps": list(FAVORABLE_PASSAGES_BPS),
        "adverse_passages_bps": list(ADVERSE_PASSAGES_BPS),
        "frozen_near_bps": FROZEN_NEAR_BPS,
        "outer_band_bps": OUTER_BAND_BPS,
        "max_book_age_ms": MAX_BOOK_AGE_MS,
        "candidate_feature_contract": "candidate_inputs.parquet + candidate_mechanics.parquet are causal by their row decision timestamp; future outcomes are isolated in candidate_decision_outcomes.parquet and candidate_7i1_outcomes.parquet",
    })
    atomic_write_json(args.output_root / "metadata" / "approved_candidate_manifest.json", approved_manifest)

    print("[7/8] Writing reports...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(coverage, args.report_root / "participant_interaction_coverage.csv")
    write_csv_atomic(day_quality, args.report_root / "participant_interaction_data_quality_by_day.csv")
    write_csv_atomic(schema_report, args.report_root / "participant_interaction_schema.csv")
    write_csv_atomic(checks, args.report_root / "participant_interaction_invariants.csv")
    atomic_write_text(args.report_root / "participant_interaction_findings.md", build_findings(summary, coverage))

    print("[8/8] Phase 7I.4 summary")
    print(f"      status={status}")
    print(f"      candidates={len(candidates):,} mechanics={len(mechanics):,} decision_outcomes={len(decision_outcomes):,}")
    p = coverage[coverage["observation_horizon_sec"] == PRIMARY_HORIZON_SEC]
    if len(p):
        print(f"      primary 3s book coverage={float(p.iloc[0]['book_coverage_rate']):.3%}")
    print(f"      reports={args.report_root}")
    if failed.empty:
        if partial_scope:
            print("Partial smoke scope completed. Do not analyze continuation probabilities until the full approved candidate population is materialized.")
        else:
            print("Full causal participant-interaction substrate completed. Freeze these datasets; subsequent continuation/economic analyses should not replay raw depth.")
        return 0
    print("ERROR: one or more Phase-7I.4 invariants failed; do not interpret downstream interaction outcomes.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
