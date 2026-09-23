#!/usr/bin/env python3
"""Phase 7M.9A — Outcome-blind Binance spot confirmation feasibility audit.

Research question
-----------------
After freezing the perp-only research branch through Phase 7M.8, can Binance
spot provide a genuinely independent causal confirmation signal for the same
prospective 25-bp BTCUSDT perp candidate population?

This phase deliberately uses SPOT TRADES ONLY.

Why spot book snapshots are excluded
------------------------------------
The canonical spot ``book_snapshots`` family is produced by the same LocalBook
implementation used by the perp collector. Earlier perp depth research showed
that this LocalBook could be invalid/unusable for material portions of some
sessions. Therefore Phase 7M.9A does NOT treat spot LocalBook snapshots as a
trusted research input. The book branch is explicitly deferred pending a
separate integrity audit.

Causal decision point
---------------------
For each Phase 7M.6F candidate with a valid 3-second anchor:

    decision_ts_recv_ns = anchor_ts_recv_ns + 3 seconds

Every spot trade admitted by this audit must satisfy ``ts_recv_ns <=`` the
relevant causal target timestamp. No future interpolation is allowed.

Families audited (no outcomes, no matching)
-------------------------------------------
1. Spot price confirmation
   * trailing 5/15/30/60-second trade-price return
   * direction-aligned return = candidate_direction * spot_return_bps
   * each price endpoint must contain a trade in the immediately preceding 1-second causal bin
   * endpoint staleness and realized span are recorded explicitly

2. Spot aggressive-flow confirmation
   * aggressive buy/sell quote from Binance aggTrade ``buyer_is_maker``
   * buyer_is_maker=False -> aggressive buy
   * buyer_is_maker=True  -> aggressive sell
   * direction-aligned net aggressive quote
   * direction-aligned trade-flow imbalance
   * primary flow feasibility requires >=5 known-side spot trades in-window

This script is FEASIBILITY ONLY. It opens no future outcomes, performs no
matching, selects no outcome-facing threshold, and creates no composite signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0.1"
PHASE = "7M.9A"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
AUDIT_WINDOWS_SEC = (5, 15, 30, 60)
PRICE_ENDPOINT_BIN_SEC = 1
MIN_KNOWN_SIDE_TRADES_FOR_FLOW = 5
NS_PER_SEC = 1_000_000_000
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_exact_participant_interaction_v1"
)
DEFAULT_SEQUENCE_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_refill_redepletion_sequence_v2"
)
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_binance_spot_confirmation_feasibility_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_binance_spot_confirmation_feasibility"
)

RAW_SPOT_TRADE_COLUMNS = (
    "ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker",
    "exchange_event_ms", "trade_time_ms", "aggregate_trade_id",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "outcome", "future", "reach_", "p75", "p50", "p100",
    "h300_", "h600_", "h900_", "h1800_", "success",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else math.nan
    except Exception:
        return math.nan


def pct(x: float) -> float:
    return float(x) * 100.0


def qstats(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    keys = ("min", "p10", "p25", "median", "p75", "p90", "p99", "max")
    if len(a) == 0:
        return {k: math.nan for k in keys}
    qs = np.quantile(a, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    return dict(zip(keys, map(float, qs)))


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def forbid_outcome_columns(frame: pd.DataFrame, context: str) -> None:
    bad = [
        c for c in frame.columns
        if any(tok in str(c).lower() for tok in FORBIDDEN_OUTCOME_TOKENS)
    ]
    if bad:
        raise RuntimeError(f"{context}: future/outcome-like columns are forbidden: {sorted(bad)}")


def read_existing_columns(path: Path, requested: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(requested))
    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    use = [c for c in requested if c in present]
    if not use:
        return pd.DataFrame(columns=list(requested))
    out = pq.read_table(path, columns=use).to_pandas()
    for c in requested:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(requested)]


# ---------------------------------------------------------------------------
# Frozen candidate clock
# ---------------------------------------------------------------------------


def load_candidate_inputs(root: Path) -> pd.DataFrame:
    path = root / "candidate_inputs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen candidate inputs: {path}")
    cols = [
        "candidate_id", "candidate_date", "direction", "direction_label",
        "candidate_second", "candidate_price", "candidate_displacement_bps",
        "policy_id", "early_threshold_bps",
    ]
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    use = [c for c in cols if c in schema]
    frame = pq.read_table(path, columns=use).to_pandas()
    forbid_outcome_columns(frame, "candidate_inputs")
    required = {"candidate_id", "candidate_date", "direction", "candidate_second"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen candidate inputs missing {sorted(missing)}")
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("Frozen candidate inputs contain duplicate candidate_id")
    if len(frame) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_CANDIDATES:,} candidates; found {len(frame):,}")
    frame = frame.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["candidate_date"] = frame["candidate_date"].astype(str)
    frame["direction"] = pd.to_numeric(frame["direction"], errors="raise").astype(int)
    frame["candidate_second"] = pd.to_numeric(frame["candidate_second"], errors="raise").astype("int64")
    if not set(frame["direction"].unique()).issubset({-1, 1}):
        raise RuntimeError("Candidate direction must be +/-1")
    return frame.sort_values(["candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True)


def load_primary_anchor_rows(sequence_root: Path) -> pd.DataFrame:
    paths = sorted(sequence_root.glob("candidate_metrics/candidate_date=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(
            "No Phase 7M.6F candidate_metrics partitions found under "
            f"{sequence_root / 'candidate_metrics'}"
        )
    requested = [
        "candidate_id", "candidate_date", "observation_horizon_sec",
        "anchor_found", "anchor_ts_recv_ns", "horizon_complete", "invalid_reason",
    ]
    parts: list[pd.DataFrame] = []
    for path in paths:
        f = read_existing_columns(path, requested)
        if f.empty:
            continue
        h = pd.to_numeric(f["observation_horizon_sec"], errors="coerce")
        f = f[h == PRIMARY_HORIZON_SEC].copy()
        if not f.empty:
            parts.append(f)
    if not parts:
        raise RuntimeError("No 3-second Phase 7M.6F candidate metric rows found")
    out = pd.concat(parts, ignore_index=True)
    out["candidate_id"] = out["candidate_id"].astype(str)
    if out["candidate_id"].duplicated().any():
        dup = out.loc[out["candidate_id"].duplicated(), "candidate_id"].head().tolist()
        raise RuntimeError(f"Duplicate primary anchor rows: {dup}")
    out["anchor_ts_recv_ns"] = pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce")
    forbid_outcome_columns(out, "7M.6F primary anchor rows")
    return out


def build_causal_candidate_clock(candidates: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in (
            "candidate_id", "anchor_found", "anchor_ts_recv_ns",
            "horizon_complete", "invalid_reason",
        ) if c in anchors.columns
    ]
    out = candidates.merge(anchors[cols], on="candidate_id", how="left", validate="one_to_one")
    anchor_found = out.get("anchor_found", False)
    if not isinstance(anchor_found, pd.Series):
        anchor_found = pd.Series(False, index=out.index)
    anchor_ok = anchor_found.fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce")
    )
    out["anchor_valid_3s"] = anchor_ok
    out["decision_ts_recv_ns"] = np.where(
        anchor_ok,
        pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce") + PRIMARY_HORIZON_SEC * NS_PER_SEC,
        np.nan,
    )
    return out


# ---------------------------------------------------------------------------
# Binance spot trade loading
# ---------------------------------------------------------------------------


def spot_trade_files(canonical_root: Path, day: date) -> list[Path]:
    base = canonical_root / "spot" / "trades" / f"date={day.isoformat()}"
    if base.exists():
        return sorted(base.rglob("*.parquet"))
    family = canonical_root / "spot" / "trades"
    if not family.exists():
        return []
    needle = f"date={day.isoformat()}"
    return sorted(p for p in family.rglob("*.parquet") if needle in p.as_posix())


@dataclass
class SpotTradeArrays:
    ts_ns: np.ndarray
    price: np.ndarray
    quote: np.ndarray
    is_buy: np.ndarray
    known_side: np.ndarray


def empty_spot_arrays() -> SpotTradeArrays:
    return SpotTradeArrays(
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        np.asarray([], dtype=bool),
        np.asarray([], dtype=bool),
    )


def inspect_spot_trade_schema(canonical_root: Path, days: Sequence[date]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for d in days:
        files = spot_trade_files(canonical_root, d)
        if not files:
            rows.append({
                "date": d.isoformat(), "file_count": 0, "row_count": 0,
                "schema_columns": "", "required_schema_ok": False,
            })
            continue
        row_count = 0
        columns: set[str] = set()
        for path in files:
            if path in seen:
                continue
            seen.add(path)
            pf = pq.ParquetFile(path)
            row_count += int(pf.metadata.num_rows)
            columns.update(pf.schema_arrow.names)
        required = {"ts_recv_ns", "price", "quantity", "buyer_is_maker"}
        rows.append({
            "date": d.isoformat(),
            "file_count": len(files),
            "row_count": row_count,
            "schema_columns": "|".join(sorted(columns)),
            "required_schema_ok": required.issubset(columns),
            "has_symbol": "symbol" in columns,
            "has_exchange_event_ms": "exchange_event_ms" in columns,
            "has_trade_time_ms": "trade_time_ms" in columns,
        })
    return pd.DataFrame(rows)


def load_spot_trade_arrays(canonical_root: Path, symbol: str, days: Sequence[date]) -> SpotTradeArrays:
    ts_parts: list[np.ndarray] = []
    px_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    buy_parts: list[np.ndarray] = []
    known_parts: list[np.ndarray] = []
    seen: set[Path] = set()

    for d in days:
        for path in spot_trade_files(canonical_root, d):
            if path in seen:
                continue
            seen.add(path)
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            required = {"ts_recv_ns", "price", "quantity", "buyer_is_maker"}
            missing = required - names
            if missing:
                raise RuntimeError(f"Spot trade file {path} missing required columns {sorted(missing)}")
            use = [c for c in RAW_SPOT_TRADE_COLUMNS if c in names]
            for batch in pf.iter_batches(batch_size=262144, columns=use):
                f = batch.to_pandas()
                if "symbol" in f.columns:
                    f = f[f["symbol"].astype(str).str.upper() == symbol.upper()]
                if f.empty:
                    continue
                ts = pd.to_numeric(f["ts_recv_ns"], errors="coerce")
                px = pd.to_numeric(f["price"], errors="coerce")
                qty = pd.to_numeric(f["quantity"], errors="coerce")
                good = np.isfinite(ts) & np.isfinite(px) & np.isfinite(qty) & (px > 0) & (qty >= 0)
                if not good.any():
                    continue
                f = f.loc[good].copy()
                ts_a = pd.to_numeric(f["ts_recv_ns"], errors="raise").astype("int64").to_numpy()
                px_a = pd.to_numeric(f["price"], errors="raise").to_numpy(float)
                qty_a = pd.to_numeric(f["quantity"], errors="raise").to_numpy(float)
                quote_a = px_a * qty_a
                maker = f["buyer_is_maker"].astype("boolean")
                known = maker.notna().to_numpy(bool)
                is_buy = maker.eq(False).fillna(False).to_numpy(bool)

                ts_parts.append(ts_a)
                px_parts.append(px_a)
                q_parts.append(quote_a)
                buy_parts.append(is_buy)
                known_parts.append(known)

    if not ts_parts:
        return empty_spot_arrays()

    ts = np.concatenate(ts_parts)
    px = np.concatenate(px_parts)
    q = np.concatenate(q_parts)
    buy = np.concatenate(buy_parts)
    known = np.concatenate(known_parts)
    order = np.argsort(ts, kind="mergesort")
    return SpotTradeArrays(ts[order], px[order], q[order], buy[order], known[order])


class SpotTradeDayCache:
    """Small LRU cache so adjacent candidate dates do not reread spot parquet."""

    def __init__(self, canonical_root: Path, symbol: str, max_days: int = 5) -> None:
        self.canonical_root = canonical_root
        self.symbol = symbol
        self.max_days = max(3, int(max_days))
        self._cache: OrderedDict[date, SpotTradeArrays] = OrderedDict()

    def get(self, day: date) -> SpotTradeArrays:
        if day in self._cache:
            arr = self._cache.pop(day)
            self._cache[day] = arr
            return arr
        arr = load_spot_trade_arrays(self.canonical_root, self.symbol, [day])
        self._cache[day] = arr
        while len(self._cache) > self.max_days:
            self._cache.popitem(last=False)
        return arr


def combine_spot_arrays(parts: Sequence[SpotTradeArrays]) -> SpotTradeArrays:
    nonempty = [x for x in parts if len(x.ts_ns) > 0]
    if not nonempty:
        return empty_spot_arrays()
    ts = np.concatenate([x.ts_ns for x in nonempty])
    px = np.concatenate([x.price for x in nonempty])
    q = np.concatenate([x.quote for x in nonempty])
    buy = np.concatenate([x.is_buy for x in nonempty])
    known = np.concatenate([x.known_side for x in nonempty])
    order = np.argsort(ts, kind="mergesort")
    return SpotTradeArrays(ts[order], px[order], q[order], buy[order], known[order])


# ---------------------------------------------------------------------------
# Exact causal spot features
# ---------------------------------------------------------------------------


def latest_trade_in_prior_bin(
    arr: SpotTradeArrays,
    target_ns: int,
    bin_sec: int = PRICE_ENDPOINT_BIN_SEC,
) -> tuple[int, float] | None:
    """Latest causal trade in the immediately preceding endpoint bin.

    For target T and bin width B, the admissible interval is (T-B, T].
    This deliberately forbids indefinite last-price carry-forward across data gaps.
    """
    if len(arr.ts_ns) == 0:
        return None
    target_ns = int(target_ns)
    lower_ns = target_ns - int(bin_sec) * NS_PER_SEC
    idx = int(np.searchsorted(arr.ts_ns, target_ns, side="right") - 1)
    if idx < 0:
        return None
    ts = int(arr.ts_ns[idx])
    if ts <= lower_ns:
        return None
    return ts, float(arr.price[idx])


def summarize_spot_window(
    arr: SpotTradeArrays,
    decision_ns: int,
    direction: int,
    window_sec: int,
) -> dict[str, float]:
    if len(arr.ts_ns) == 0:
        return {
            "price_return_bps": math.nan,
            "directional_price_return_bps": math.nan,
            "price_now_age_sec": math.nan,
            "price_prior_age_sec": math.nan,
            "price_realized_span_sec": math.nan,
            "price_now_endpoint_bin_valid": 0.0,
            "price_prior_endpoint_bin_valid": 0.0,
            "trade_count": 0.0,
            "known_side_trade_count": 0.0,
            "unknown_side_trade_count": 0.0,
            "total_quote": 0.0,
            "known_side_quote": 0.0,
            "aggressive_buy_quote": 0.0,
            "aggressive_sell_quote": 0.0,
            "net_aggressive_quote": 0.0,
            "directional_net_aggressive_quote": 0.0,
            "trade_flow_imbalance": math.nan,
            "directional_trade_flow_imbalance": math.nan,
            "directional_aggressive_quote": 0.0,
            "counter_aggressive_quote": 0.0,
            "flow_valid_ge5_known_side": 0.0,
        }

    start_target_ns = int(decision_ns - window_sec * NS_PER_SEC)
    now = latest_trade_in_prior_bin(arr, int(decision_ns))
    prior = latest_trade_in_prior_bin(arr, start_target_ns)

    if now is not None and prior is not None and prior[1] > 0:
        ret_bps = 10_000.0 * (now[1] / prior[1] - 1.0)
        now_age = (int(decision_ns) - now[0]) / NS_PER_SEC
        prior_age = (start_target_ns - prior[0]) / NS_PER_SEC
        realized_span = (now[0] - prior[0]) / NS_PER_SEC
    else:
        ret_bps = math.nan
        now_age = math.nan if now is None else (int(decision_ns) - now[0]) / NS_PER_SEC
        prior_age = math.nan if prior is None else (start_target_ns - prior[0]) / NS_PER_SEC
        realized_span = math.nan if now is None or prior is None else (now[0] - prior[0]) / NS_PER_SEC

    lo = int(np.searchsorted(arr.ts_ns, start_target_ns, side="right"))
    hi = int(np.searchsorted(arr.ts_ns, int(decision_ns), side="right"))
    if hi <= lo:
        return {
            "price_return_bps": ret_bps,
            "directional_price_return_bps": float(direction) * ret_bps if math.isfinite(ret_bps) else math.nan,
            "price_now_age_sec": now_age,
            "price_prior_age_sec": prior_age,
            "price_realized_span_sec": realized_span,
            "price_now_endpoint_bin_valid": float(now is not None),
            "price_prior_endpoint_bin_valid": float(prior is not None),
            "trade_count": 0.0,
            "known_side_trade_count": 0.0,
            "unknown_side_trade_count": 0.0,
            "total_quote": 0.0,
            "known_side_quote": 0.0,
            "aggressive_buy_quote": 0.0,
            "aggressive_sell_quote": 0.0,
            "net_aggressive_quote": 0.0,
            "directional_net_aggressive_quote": 0.0,
            "trade_flow_imbalance": math.nan,
            "directional_trade_flow_imbalance": math.nan,
            "directional_aggressive_quote": 0.0,
            "counter_aggressive_quote": 0.0,
            "flow_valid_ge5_known_side": 0.0,
        }

    quote = arr.quote[lo:hi]
    known = arr.known_side[lo:hi]
    buy = arr.is_buy[lo:hi]
    known_quote = quote[known]
    buy_quote = float(np.sum(quote[known & buy]))
    sell_quote = float(np.sum(quote[known & ~buy]))
    total_quote = float(np.sum(quote))
    known_total = float(np.sum(known_quote))
    net = buy_quote - sell_quote
    imbalance = net / known_total if known_total > 0 else math.nan
    directional_net = float(direction) * net
    directional_imbalance = float(direction) * imbalance if math.isfinite(imbalance) else math.nan
    directional_aggr = buy_quote if int(direction) > 0 else sell_quote
    counter_aggr = sell_quote if int(direction) > 0 else buy_quote

    return {
        "price_return_bps": ret_bps,
        "directional_price_return_bps": float(direction) * ret_bps if math.isfinite(ret_bps) else math.nan,
        "price_now_age_sec": now_age,
        "price_prior_age_sec": prior_age,
        "price_realized_span_sec": realized_span,
        "price_now_endpoint_bin_valid": float(now is not None),
        "price_prior_endpoint_bin_valid": float(prior is not None),
        "trade_count": float(hi - lo),
        "known_side_trade_count": float(np.sum(known)),
        "unknown_side_trade_count": float(np.sum(~known)),
        "total_quote": total_quote,
        "known_side_quote": known_total,
        "aggressive_buy_quote": buy_quote,
        "aggressive_sell_quote": sell_quote,
        "net_aggressive_quote": net,
        "directional_net_aggressive_quote": directional_net,
        "trade_flow_imbalance": imbalance,
        "directional_trade_flow_imbalance": directional_imbalance,
        "directional_aggressive_quote": directional_aggr,
        "counter_aggressive_quote": counter_aggr,
        "flow_valid_ge5_known_side": float(np.sum(known) >= MIN_KNOWN_SIDE_TRADES_FOR_FLOW),
    }


def attach_spot_trade_features(
    candidates: pd.DataFrame,
    canonical_root: Path,
    symbol: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = candidates[candidates["anchor_valid_3s"]].copy()
    cache = SpotTradeDayCache(canonical_root, symbol, max_days=5)
    groups = list(valid.groupby("candidate_date", sort=True))
    for idx, (day_s, group) in enumerate(groups, start=1):
        center = date.fromisoformat(str(day_s))
        arr = combine_spot_arrays([
            cache.get(center - timedelta(days=1)),
            cache.get(center),
            cache.get(center + timedelta(days=1)),
        ])
        print(
            f"      [{idx:02d}/{len(groups):02d}] {day_s}: "
            f"candidates={len(group):3d} context_trades={len(arr.ts_ns):,}"
        )
        for r in group.itertuples(index=False):
            rec: dict[str, Any] = {
                "candidate_id": r.candidate_id,
                "spot_trade_files_available": bool(len(arr.ts_ns) > 0),
            }
            if len(arr.ts_ns) > 0:
                decision_ns = int(r.decision_ts_recv_ns)
                for w in AUDIT_WINDOWS_SEC:
                    s = summarize_spot_window(arr, decision_ns, int(r.direction), w)
                    for key, value in s.items():
                        rec[f"spot_{key}_{w}s"] = value
            rows.append(rec)

    aux = pd.DataFrame(rows)
    if aux.empty:
        return candidates.copy()
    return candidates.merge(aux, on="candidate_id", how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# Feasibility summaries
# ---------------------------------------------------------------------------


def summarize_price_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    anchored = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    for w in AUDIT_WINDOWS_SEC:
        dr = pd.to_numeric(anchored.get(f"spot_directional_price_return_bps_{w}s"), errors="coerce")
        now_age = pd.to_numeric(anchored.get(f"spot_price_now_age_sec_{w}s"), errors="coerce")
        prior_age = pd.to_numeric(anchored.get(f"spot_price_prior_age_sec_{w}s"), errors="coerce")
        span = pd.to_numeric(anchored.get(f"spot_price_realized_span_sec_{w}s"), errors="coerce")
        now_valid = pd.to_numeric(anchored.get(f"spot_price_now_endpoint_bin_valid_{w}s"), errors="coerce").fillna(0) > 0
        prior_valid = pd.to_numeric(anchored.get(f"spot_price_prior_endpoint_bin_valid_{w}s"), errors="coerce").fillna(0) > 0
        finite = np.isfinite(dr) & now_valid & prior_valid
        pos = finite & (dr > 0)
        neg = finite & (dr < 0)
        zero = finite & (np.abs(dr) <= 1e-15)
        rows.append({
            "window_sec": w,
            "anchored_candidates": len(anchored),
            "finite_candidates": int(finite.sum()),
            "finite_share_pct": pct(float(finite.mean())) if len(anchored) else math.nan,
            "endpoint_bin_width_sec": PRICE_ENDPOINT_BIN_SEC,
            "now_endpoint_bin_valid_count": int(now_valid.sum()),
            "prior_endpoint_bin_valid_count": int(prior_valid.sum()),
            "both_endpoint_bins_valid_count": int((now_valid & prior_valid).sum()),
            "direction_aligned_positive_count": int(pos.sum()),
            "direction_aligned_positive_share_pct": pct(float(pos.mean())) if len(anchored) else math.nan,
            "direction_aligned_negative_count": int(neg.sum()),
            "direction_aligned_negative_share_pct": pct(float(neg.mean())) if len(anchored) else math.nan,
            "exact_zero_share_pct": pct(float(zero.mean())) if len(anchored) else math.nan,
            **{f"directional_return_bps_{k}": v for k, v in qstats(dr[finite]).items()},
            **{f"now_endpoint_age_sec_{k}": v for k, v in qstats(now_age[np.isfinite(now_age)]).items()},
            **{f"prior_endpoint_age_sec_{k}": v for k, v in qstats(prior_age[np.isfinite(prior_age)]).items()},
            **{f"realized_span_sec_{k}": v for k, v in qstats(span[np.isfinite(span)]).items()},
        })
    return pd.DataFrame(rows)


def summarize_flow_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    anchored = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    for w in AUDIT_WINDOWS_SEC:
        net = pd.to_numeric(anchored.get(f"spot_directional_net_aggressive_quote_{w}s"), errors="coerce")
        imb = pd.to_numeric(anchored.get(f"spot_directional_trade_flow_imbalance_{w}s"), errors="coerce")
        count = pd.to_numeric(anchored.get(f"spot_trade_count_{w}s"), errors="coerce")
        known_count = pd.to_numeric(anchored.get(f"spot_known_side_trade_count_{w}s"), errors="coerce")
        dq = pd.to_numeric(anchored.get(f"spot_directional_aggressive_quote_{w}s"), errors="coerce")
        cq = pd.to_numeric(anchored.get(f"spot_counter_aggressive_quote_{w}s"), errors="coerce")
        finite = np.isfinite(net)
        flow_valid = finite & np.isfinite(known_count) & (known_count >= MIN_KNOWN_SIDE_TRADES_FOR_FLOW)
        pos = flow_valid & (net > 0)
        neg = flow_valid & (net < 0)
        zero = flow_valid & (np.abs(net) <= 1e-9)
        rows.append({
            "window_sec": w,
            "anchored_candidates": len(anchored),
            "finite_candidates": int(finite.sum()),
            "finite_share_pct": pct(float(finite.mean())) if len(anchored) else math.nan,
            "primary_eligible_candidates": int(flow_valid.sum()),
            "primary_eligible_share_pct": pct(float(flow_valid.mean())) if len(anchored) else math.nan,
            "minimum_known_side_trades": MIN_KNOWN_SIDE_TRADES_FOR_FLOW,
            "direction_aligned_net_positive_count": int(pos.sum()),
            "direction_aligned_net_positive_share_pct": pct(float(pos.mean())) if len(anchored) else math.nan,
            "direction_aligned_net_positive_share_of_eligible_pct": pct(float(pos.sum() / flow_valid.sum())) if flow_valid.sum() else math.nan,
            "direction_aligned_net_negative_count": int(neg.sum()),
            "direction_aligned_net_negative_share_pct": pct(float(neg.mean())) if len(anchored) else math.nan,
            "direction_aligned_net_negative_share_of_eligible_pct": pct(float(neg.sum() / flow_valid.sum())) if flow_valid.sum() else math.nan,
            "exact_zero_net_share_pct": pct(float(zero.mean())) if len(anchored) else math.nan,
            "trade_count_ge5_share_pct": pct(float((count >= 5).mean())) if len(anchored) else math.nan,
            "known_side_trade_count_ge5_share_pct": pct(float((known_count >= 5).mean())) if len(anchored) else math.nan,
            **{f"directional_net_quote_{k}": v for k, v in qstats(net[flow_valid]).items()},
            **{f"directional_flow_imbalance_{k}": v for k, v in qstats(imb[flow_valid & np.isfinite(imb)]).items()},
            **{f"directional_aggressive_quote_{k}": v for k, v in qstats(dq[np.isfinite(dq)]).items()},
            **{f"counter_aggressive_quote_{k}": v for k, v in qstats(cq[np.isfinite(cq)]).items()},
            **{f"trade_count_{k}": v for k, v in qstats(count[np.isfinite(count)]).items()},
        })
    return pd.DataFrame(rows)


def subgroup_representation(frame: pd.DataFrame) -> pd.DataFrame:
    anchored = frame[frame["anchor_valid_3s"]].copy()
    price5 = pd.to_numeric(anchored.get("spot_directional_price_return_bps_5s"), errors="coerce")
    price30 = pd.to_numeric(anchored.get("spot_directional_price_return_bps_30s"), errors="coerce")
    flow30 = pd.to_numeric(anchored.get("spot_directional_net_aggressive_quote_30s"), errors="coerce")
    known30 = pd.to_numeric(anchored.get("spot_known_side_trade_count_30s"), errors="coerce")
    flow_valid30 = np.isfinite(flow30) & np.isfinite(known30) & (known30 >= MIN_KNOWN_SIDE_TRADES_FOR_FLOW)
    rows: list[dict[str, Any]] = []
    masks = [
        ("FULL", pd.Series(True, index=anchored.index)),
        ("DISCOVERY", anchored["candidate_date"].astype(str) <= DISCOVERY_END_DATE),
        ("VALIDATION", anchored["candidate_date"].astype(str) >= VALIDATION_START_DATE),
        ("UP", pd.to_numeric(anchored["direction"], errors="coerce") > 0),
        ("DOWN", pd.to_numeric(anchored["direction"], errors="coerce") < 0),
    ]
    for label, mask in masks:
        p5 = price5[mask]
        p30 = price30[mask]
        f30 = flow30[mask]
        fv30 = flow_valid30[mask]
        rows.append({
            "subgroup": label,
            "candidate_count": int(mask.sum()),
            "finite_price_5s": int(np.isfinite(p5).sum()),
            "price_aligned_5s": int((np.isfinite(p5) & (p5 > 0)).sum()),
            "price_counter_5s": int((np.isfinite(p5) & (p5 < 0)).sum()),
            "finite_price_30s": int(np.isfinite(p30).sum()),
            "price_aligned_30s": int((np.isfinite(p30) & (p30 > 0)).sum()),
            "price_counter_30s": int((np.isfinite(p30) & (p30 < 0)).sum()),
            "flow_primary_eligible_30s": int(fv30.sum()),
            "flow_aligned_30s": int((fv30 & (f30 > 0)).sum()),
            "flow_counter_30s": int((fv30 & (f30 < 0)).sum()),
        })
    return pd.DataFrame(rows)


def family_availability(frame: pd.DataFrame) -> pd.DataFrame:
    anchored = frame[frame["anchor_valid_3s"]].copy()
    price5 = pd.to_numeric(anchored.get("spot_directional_price_return_bps_5s"), errors="coerce")
    price30 = pd.to_numeric(anchored.get("spot_directional_price_return_bps_30s"), errors="coerce")
    flow30 = pd.to_numeric(anchored.get("spot_directional_net_aggressive_quote_30s"), errors="coerce")
    known30 = pd.to_numeric(anchored.get("spot_known_side_trade_count_30s"), errors="coerce")
    flow_valid30 = np.isfinite(flow30) & np.isfinite(known30) & (known30 >= MIN_KNOWN_SIDE_TRADES_FOR_FLOW)
    rows = [
        {"family": "ANCHOR_3S", "usable": int(frame["anchor_valid_3s"].sum()), "total": len(frame)},
        {"family": "SPOT_PRICE_CONFIRMATION_5S_STRICT_ENDPOINT", "usable": int(np.isfinite(price5).sum()), "total": len(anchored)},
        {"family": "SPOT_PRICE_CONFIRMATION_30S_STRICT_ENDPOINT", "usable": int(np.isfinite(price30).sum()), "total": len(anchored)},
        {"family": "SPOT_AGGRESSIVE_FLOW_30S_GE5_KNOWN_SIDE", "usable": int(flow_valid30.sum()), "total": len(anchored)},
    ]
    for r in rows:
        r["usable_share_pct"] = pct(r["usable"] / r["total"]) if r["total"] else math.nan
    return pd.DataFrame(rows)


def make_findings(frame: pd.DataFrame, price: pd.DataFrame, flow: pd.DataFrame) -> str:
    anchored = int(frame["anchor_valid_3s"].sum())
    p5 = price.loc[price["window_sec"] == 5].iloc[0].to_dict()
    p30 = price.loc[price["window_sec"] == 30].iloc[0].to_dict()
    f30 = flow.loc[flow["window_sec"] == 30].iloc[0].to_dict()
    return f"""# Phase 7M.9A v1.0.1 — Binance spot confirmation feasibility

Status: **OUTCOME-BLIND FEASIBILITY AUDIT COMPLETE — STRICT PRICE-ENDPOINT INTEGRITY**

## Frozen scope

- Prospective 25-bp perp candidates: **{len(frame):,}**
- Exact 7M.6F anchor +3s decision points available: **{anchored:,}** ({anchored/len(frame)*100:.2f}%)
- Future outcome tables opened: **NO**
- Matching performed: **NO**
- Composite signal built: **NO**
- Spot LocalBook snapshots used: **NO**

## Spot price integrity correction

Price endpoints are no longer indefinitely carried forward. For each target T, the endpoint price must be the latest causal spot trade in **(T-1s, T]**. If that one-second bin is empty, the endpoint — and therefore the return — is unavailable.

### 5-second spot price confirmation (candidate for 7M.9B secondary test)

- Strict finite causal coverage: **{p5['finite_share_pct']:.2f}%**
- Direction-aligned positive return share: **{p5['direction_aligned_positive_share_pct']:.2f}%**
- Direction-counter negative return share: **{p5['direction_aligned_negative_share_pct']:.2f}%**
- Median directional spot return: **{p5['directional_return_bps_median']:.3f} bp**

### 30-second spot price confirmation (descriptive only unless representation changes materially)

- Strict finite causal coverage: **{p30['finite_share_pct']:.2f}%**
- Direction-aligned positive return share: **{p30['direction_aligned_positive_share_pct']:.2f}%**
- Direction-counter negative return share: **{p30['direction_aligned_negative_share_pct']:.2f}%**
- Median decision-endpoint staleness: **{p30['now_endpoint_age_sec_median']:.4f}s**
- Median earlier-endpoint staleness: **{p30['prior_endpoint_age_sec_median']:.4f}s**

## 30-second spot aggressive-flow confirmation (candidate for 7M.9B primary test)

- Raw finite causal coverage: **{f30['finite_share_pct']:.2f}%**
- Primary eligible (>= {MIN_KNOWN_SIDE_TRADES_FOR_FLOW} known-side spot trades): **{f30['primary_eligible_share_pct']:.2f}%**
- Direction-aligned net aggressive-flow positive share: **{f30['direction_aligned_net_positive_share_pct']:.2f}%**
- Direction-counter net aggressive-flow negative share: **{f30['direction_aligned_net_negative_share_pct']:.2f}%**
- Median direction-aligned trade-flow imbalance: **{f30['directional_flow_imbalance_median']:.6f}**

Interpretation: flow is a direct aggregation of observed executions in (T-30s, T]; it is not forward-filled and is independent of the price-endpoint correction.

## Spot book branch

**DEFERRED / NOT USED.** Canonical spot book snapshots come from the same LocalBook implementation whose integrity failed for material stretches of earlier perp depth research.

## Next gate

If strict 5-second price counter-confirmation and 30-second flow counter-confirmation retain sufficient discovery/validation representation, Phase 7M.9B may freeze them as separate matching hypotheses. No P75 outcomes have been opened here.
"""



# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    arr = SpotTradeArrays(
        ts_ns=np.asarray([1, 2, 3, 4, 5], dtype=np.int64) * NS_PER_SEC,
        price=np.asarray([100.0, 101.0, 102.0, 103.0, 104.0]),
        quote=np.asarray([100.0, 200.0, 300.0, 400.0, 500.0]),
        is_buy=np.asarray([True, False, True, False, True]),
        known_side=np.asarray([True, True, True, True, True]),
    )
    up = summarize_spot_window(arr, 4_500_000_000, +1, 3)
    # Window flow is (1.5s, 4.5s] => timestamps 2,3,4.
    assert up["trade_count"] == 3
    assert abs(up["aggressive_buy_quote"] - 300.0) < 1e-12
    assert abs(up["aggressive_sell_quote"] - 600.0) < 1e-12
    assert abs(up["directional_net_aggressive_quote"] + 300.0) < 1e-12
    # Price now=103 at 4s; prior=101 at 2s because target is 1.5s and latest <=1.5 is ts=1.
    # Correct prior is actually 100 at 1s.
    expected_ret = 10_000.0 * (103.0 / 100.0 - 1.0)
    assert abs(up["price_return_bps"] - expected_ret) < 1e-9
    assert abs(up["directional_price_return_bps"] - expected_ret) < 1e-9
    assert abs(up["price_now_age_sec"] - 0.5) < 1e-12
    assert abs(up["price_prior_age_sec"] - 0.5) < 1e-12
    assert abs(up["price_realized_span_sec"] - 3.0) < 1e-12

    down = summarize_spot_window(arr, 4_500_000_000, -1, 3)
    assert abs(down["directional_net_aggressive_quote"] - 300.0) < 1e-12
    assert down["directional_price_return_bps"] < 0

    # Strict endpoint bins forbid stale last-price carry-forward.
    stale = latest_trade_in_prior_bin(arr, 10 * NS_PER_SEC)
    assert stale is None
    exact = latest_trade_in_prior_bin(arr, 5 * NS_PER_SEC)
    assert exact is not None and exact[0] == 5 * NS_PER_SEC

    c = pd.DataFrame({
        "candidate_id": ["x"], "candidate_date": ["2026-06-01"],
        "direction": [1], "candidate_second": [100],
    })
    a = pd.DataFrame({
        "candidate_id": ["x"], "anchor_found": [True],
        "anchor_ts_recv_ns": [100_900_000_000], "horizon_complete": [True],
    })
    cc = build_causal_candidate_clock(c, a)
    assert int(cc.loc[0, "decision_ts_recv_ns"]) == 103_900_000_000

    print(
        "SELF-TEST PASS: exact anchor+3s clock, strict 1s causal spot-price endpoints, "
        "spot aggTrade aggressor semantics, >=5 known-side flow eligibility, and direction alignment validated."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    print(f"Phase {PHASE} - Outcome-blind Binance spot confirmation feasibility")
    print("[1/6] Loading frozen 7M.3 candidate universe and exact 7M.6F anchor+3s clock...")
    candidates = load_candidate_inputs(args.candidate_root)
    anchors = load_primary_anchor_rows(args.sequence_root)
    frame = build_causal_candidate_clock(candidates, anchors)
    print(f"      candidates={len(frame):,} anchor+3s={int(frame['anchor_valid_3s'].sum()):,}")

    print("[2/6] Auditing canonical Binance spot trade availability/schema...")
    unique_days = sorted({date.fromisoformat(x) for x in frame["candidate_date"].astype(str).unique()})
    schema_audit = inspect_spot_trade_schema(args.canonical_root, unique_days)
    bad_schema = schema_audit[(schema_audit["file_count"] > 0) & (~schema_audit["required_schema_ok"].astype(bool))]
    if len(bad_schema):
        raise RuntimeError(
            "Spot trade canonical schema is missing required fields on dates: "
            + ", ".join(bad_schema["date"].astype(str).tolist()[:10])
        )

    print("[3/6] Aggregating exact causal Binance spot price/flow windows (trades only; no LocalBook)...")
    frame = attach_spot_trade_features(frame, args.canonical_root, args.symbol)

    print("[4/6] Computing outcome-blind feasibility and subgroup summaries...")
    price = summarize_price_confirmation(frame)
    flow = summarize_flow_confirmation(frame)
    subgroups = subgroup_representation(frame)
    availability = family_availability(frame)

    print("[5/6] Writing causal audit substrate and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    forbid_outcome_columns(frame, "7M.9A candidate spot audit output")
    causal_path = args.output_root / "candidate_binance_spot_confirmation_audit.parquet"
    frame.to_parquet(causal_path, index=False)

    write_csv(args.report_root / "spot_trade_availability_by_day.csv", schema_audit)
    write_csv(args.report_root / "family_availability.csv", availability)
    write_csv(args.report_root / "spot_price_confirmation_feasibility.csv", price)
    write_csv(args.report_root / "spot_aggressive_flow_feasibility.csv", flow)
    write_csv(args.report_root / "subgroup_representation.csv", subgroups)

    book_deferred = pd.DataFrame([
        {
            "family": "SPOT_LOCALBOOK_SNAPSHOT_CONFIRMATION",
            "status": "DEFERRED_NOT_USED",
            "reason": (
                "same LocalBook implementation as perps; earlier perp depth research found "
                "material invalid/unusable book intervals; requires separate integrity audit"
            ),
            "canonical_family": "spot/book_snapshots",
            "used_in_7m9a": False,
        }
    ])
    write_csv(args.report_root / "deferred_spot_book_branch.csv", book_deferred)

    invariants = pd.DataFrame([
        {"invariant": "future_outcomes_opened", "value": False},
        {"invariant": "matching_performed", "value": False},
        {"invariant": "composite_signal_built", "value": False},
        {"invariant": "decision_clock", "value": "exact_7m6f_anchor_ts_recv_ns_plus_3s"},
        {"invariant": "spot_trade_causality", "value": "ts_recv_ns<=causal_target_timestamp"},
        {"invariant": "spot_price_endpoint_rule", "value": "latest_trade_in_(T-1s,T]_else_unavailable"},
        {"invariant": "spot_flow_primary_min_known_side_trades", "value": MIN_KNOWN_SIDE_TRADES_FOR_FLOW},
        {"invariant": "spot_aggressor_semantics", "value": "buyer_is_maker_false=aggressive_buy_true=aggressive_sell"},
        {"invariant": "spot_book_snapshots_used", "value": False},
        {"invariant": "spot_book_status", "value": "DEFERRED_PENDING_INTEGRITY_AUDIT"},
        {"invariant": "raw_spot_depth_replay", "value": False},
        {"invariant": "candidate_count", "value": len(frame)},
        {"invariant": "anchor_valid_3s_count", "value": int(frame["anchor_valid_3s"].sum())},
    ])
    write_csv(args.report_root / "analysis_invariants.csv", invariants)

    findings = make_findings(frame, price, flow)
    (args.report_root / "findings.md").write_text(findings, encoding="utf-8")

    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "symbol": args.symbol,
        "candidate_count": len(frame),
        "anchor_valid_3s_count": int(frame["anchor_valid_3s"].sum()),
        "windows_sec": list(AUDIT_WINDOWS_SEC),
        "price_endpoint_bin_sec": PRICE_ENDPOINT_BIN_SEC,
        "flow_primary_min_known_side_trades": MIN_KNOWN_SIDE_TRADES_FOR_FLOW,
        "future_outcomes_opened": False,
        "matching_performed": False,
        "composite_signal_built": False,
        "spot_book_snapshots_used": False,
        "spot_book_status": "DEFERRED_PENDING_INTEGRITY_AUDIT",
        "candidate_input_sha256": sha256_file(args.candidate_root / "candidate_inputs.parquet"),
        "causal_output": str(causal_path),
        "reports": {
            "spot_trade_availability": str(args.report_root / "spot_trade_availability_by_day.csv"),
            "family_availability": str(args.report_root / "family_availability.csv"),
            "spot_price_confirmation": str(args.report_root / "spot_price_confirmation_feasibility.csv"),
            "spot_aggressive_flow": str(args.report_root / "spot_aggressive_flow_feasibility.csv"),
            "subgroups": str(args.report_root / "subgroup_representation.csv"),
            "deferred_spot_book": str(args.report_root / "deferred_spot_book_branch.csv"),
            "findings": str(args.report_root / "findings.md"),
        },
    }
    write_json(args.report_root / "summary.json", summary)

    print("[6/6] Complete.")
    print(f"      causal={causal_path}")
    print(f"      report={args.report_root}")
    print("      outcomes_opened=NO matching=NO composite=NO spot_book=DEFERRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
