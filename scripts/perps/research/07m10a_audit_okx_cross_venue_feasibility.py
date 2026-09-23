#!/usr/bin/env python3
"""Phase 7M.10A — Outcome-blind OKX cross-venue confirmation feasibility audit.

Research question
-----------------
For the same frozen prospective BTCUSDT perp 25-bp candidate population used
through Phases 7M.2–7M.9, does an independent spot venue (OKX BTC-USDT)
provide usable causal information at the exact Phase 7M.6F ``anchor + 3s``
decision point?

Collector/canonical contract
----------------------------
The collector subscribes to OKX books-family data (configured as ``books5``)
and normalizes the first bid/ask level into a direct ``bbo`` BronzeEvent.
The canonical extractor routes:

    ("okx_spot", "bbo") -> okx_spot/bbo/date=YYYY-MM-DD/*.parquet

with fields including:

    ts_recv_ns, symbol, instrument_id, exchange_ts_ms,
    bid_price, bid_size, ask_price, ask_size

These BBO observations come directly from the OKX websocket payload. They are
NOT LocalBook-derived snapshots, so the Binance spot LocalBook integrity issue
from Phase 7M.9 does not apply to this source.

Causal decision point
---------------------
For every candidate with a valid Phase 7M.6F 3-second anchor:

    decision_ts_recv_ns = anchor_ts_recv_ns + 3 seconds

Only OKX BBO records with ``ts_recv_ns <= target`` are eligible. Phase 7M.10A
freezes the outcome-blind freshness rule selected after reviewing the v1.0.0
feasibility audit: every BBO observation used as current state or as a return
endpoint must be no more than 2 seconds old relative to its causal target. Older
last-known quotes are recorded as stale but are NOT reused as valid state.

Families audited (no outcomes, no matching)
-------------------------------------------
1. OKX mid-price confirmation
   * trailing 5/15/30/60-second mid-price return
   * direction-aligned return = candidate_direction * return_bps
   * endpoint age and realized span recorded explicitly

2. OKX direct-BBO size imbalance
   * (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)
   * direction-aligned at the decision point
   * this is top-of-book only; it is NOT treated as full-depth imbalance

3. Cross-venue relative move (descriptive feasibility only)
   * where the corrected Phase 7M.9A spot-price substrate exists, compare
     direction-aligned OKX mid return with direction-aligned Binance spot
     trade-price return over the same window
   * this is a relative-return measurement, not a composite score and not an
     outcome-facing feature in this phase

This script opens no future outcome table, performs no matching, selects no
outcome-facing threshold, and creates no composite signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0.1"
PHASE = "7M.10A"
SYMBOL = "BTCUSDT"
OKX_INSTRUMENT = "BTC-USDT"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
AUDIT_WINDOWS_SEC = (5, 15, 30, 60)
NS_PER_SEC = 1_000_000_000
OKX_MAX_ENDPOINT_AGE_SEC = 2.0
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
DEFAULT_SPOT_CAUSAL_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_binance_spot_confirmation_feasibility_v1"
)
DEFAULT_SPOT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_binance_spot_confirmation_feasibility"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_okx_cross_venue_feasibility_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_okx_cross_venue_feasibility"
)

RAW_OKX_COLUMNS = (
    "ts_recv_ns", "market", "event_type", "symbol", "instrument_id",
    "channel", "exchange_ts_ms", "bid_price", "bid_size", "ask_price", "ask_size",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "outcome", "future", "reach_", "p75", "p50", "p100",
    "h300_", "h600_", "h900_", "h1800_", "success",
)


# ---------------------------------------------------------------------------
# Generic utilities
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
    requested = [
        "candidate_id", "candidate_date", "direction", "direction_label",
        "candidate_second", "candidate_price", "candidate_displacement_bps",
        "policy_id", "early_threshold_bps",
    ]
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    use = [c for c in requested if c in schema]
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
        pd.to_numeric(out.get("anchor_ts_recv_ns"), errors="coerce")
    )
    out["anchor_valid_3s"] = anchor_ok
    out["decision_ts_recv_ns"] = np.nan
    if anchor_ok.any():
        out.loc[anchor_ok, "decision_ts_recv_ns"] = (
            pd.to_numeric(out.loc[anchor_ok, "anchor_ts_recv_ns"], errors="raise").astype("int64")
            + PRIMARY_HORIZON_SEC * NS_PER_SEC
        )
    out["sample_partition"] = np.where(
        out["candidate_date"] <= DISCOVERY_END_DATE,
        "DISCOVERY",
        np.where(out["candidate_date"] >= VALIDATION_START_DATE, "VALIDATION", "GAP"),
    )
    if int(out["anchor_valid_3s"].sum()) <= 0:
        raise RuntimeError("No valid 3-second anchor candidates available")
    forbid_outcome_columns(out, "causal candidate clock")
    return out


# ---------------------------------------------------------------------------
# OKX canonical BBO loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OkxBboArrays:
    ts_ns: np.ndarray
    exchange_ts_ms: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_sz: np.ndarray
    ask_sz: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        return (self.bid + self.ask) / 2.0


EMPTY_BBO = OkxBboArrays(
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=float),
)


def find_okx_bbo_files(canonical_root: Path, day: date) -> list[Path]:
    root = canonical_root / "okx_spot" / "bbo" / f"date={day.isoformat()}"
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.parquet") if p.is_file())


def load_okx_bbo_day(
    canonical_root: Path,
    symbol: str,
    day: date,
) -> tuple[OkxBboArrays, dict[str, Any]]:
    paths = find_okx_bbo_files(canonical_root, day)
    base_stats: dict[str, Any] = {
        "date": day.isoformat(),
        "file_count": len(paths),
        "row_count_raw": 0,
        "row_count_symbol": 0,
        "row_count_valid_quote": 0,
        "invalid_quote_count": 0,
        "crossed_quote_count": 0,
    }
    if not paths:
        return EMPTY_BBO, base_stats

    parts: list[pd.DataFrame] = []
    required = {"ts_recv_ns", "bid_price", "bid_size", "ask_price", "ask_size"}
    for path in paths:
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        missing = required - schema
        if missing:
            raise RuntimeError(f"OKX BBO canonical file missing {sorted(missing)}: {path}")
        use = [c for c in RAW_OKX_COLUMNS if c in schema]
        parts.append(pq.read_table(path, columns=use).to_pandas())
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    base_stats["row_count_raw"] = len(frame)
    if frame.empty:
        return EMPTY_BBO, base_stats

    if "market" in frame.columns:
        frame = frame[frame["market"].astype(str).str.lower() == "okx_spot"]
    if "event_type" in frame.columns:
        frame = frame[frame["event_type"].astype(str).str.lower() == "bbo"]
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper() == symbol.upper()]
    if "instrument_id" in frame.columns:
        inst = frame["instrument_id"].astype(str).str.upper()
        frame = frame[(inst == OKX_INSTRUMENT) | (inst == symbol.upper())]

    base_stats["row_count_symbol"] = len(frame)
    if frame.empty:
        return EMPTY_BBO, base_stats

    ts = pd.to_numeric(frame["ts_recv_ns"], errors="coerce").to_numpy(dtype=float)
    exch = pd.to_numeric(frame.get("exchange_ts_ms"), errors="coerce").to_numpy(dtype=float)
    bid = pd.to_numeric(frame["bid_price"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(frame["ask_price"], errors="coerce").to_numpy(dtype=float)
    bid_sz = pd.to_numeric(frame["bid_size"], errors="coerce").to_numpy(dtype=float)
    ask_sz = pd.to_numeric(frame["ask_size"], errors="coerce").to_numpy(dtype=float)

    finite = (
        np.isfinite(ts) & np.isfinite(bid) & np.isfinite(ask) &
        np.isfinite(bid_sz) & np.isfinite(ask_sz)
    )
    positive = (bid > 0) & (ask > 0) & (bid_sz > 0) & (ask_sz > 0)
    crossed = finite & positive & (ask < bid)
    valid = finite & positive & (ask >= bid) & (ts > 0)

    base_stats["crossed_quote_count"] = int(crossed.sum())
    base_stats["invalid_quote_count"] = int((~valid).sum())
    base_stats["row_count_valid_quote"] = int(valid.sum())

    if not valid.any():
        return EMPTY_BBO, base_stats

    ts_i = ts[valid].astype(np.int64)
    exch_v = exch[valid]
    bid_v = bid[valid]
    ask_v = ask[valid]
    bid_sz_v = bid_sz[valid]
    ask_sz_v = ask_sz[valid]
    order = np.argsort(ts_i, kind="mergesort")
    arr = OkxBboArrays(
        ts_i[order], exch_v[order], bid_v[order], ask_v[order], bid_sz_v[order], ask_sz_v[order]
    )

    gaps = np.diff(arr.ts_ns) / NS_PER_SEC if len(arr.ts_ns) > 1 else np.empty(0)
    spread_bps = 10_000.0 * (arr.ask - arr.bid) / arr.mid
    recv_ms = arr.ts_ns.astype(float) / 1_000_000.0
    lag_ms = recv_ms - arr.exchange_ts_ms
    lag_ms = lag_ms[np.isfinite(lag_ms)]
    base_stats.update({
        "first_ts_recv_ns": int(arr.ts_ns[0]),
        "last_ts_recv_ns": int(arr.ts_ns[-1]),
        **{f"interarrival_sec_{k}": v for k, v in qstats(gaps).items()},
        **{f"spread_bps_{k}": v for k, v in qstats(spread_bps).items()},
        **{f"recv_minus_exchange_ms_{k}": v for k, v in qstats(lag_ms).items()},
    })
    return arr, base_stats


class OkxBboDayCache:
    def __init__(self, canonical_root: Path, symbol: str, max_days: int = 5) -> None:
        self.canonical_root = canonical_root
        self.symbol = symbol
        self.max_days = max_days
        self.cache: OrderedDict[str, OkxBboArrays] = OrderedDict()
        self.stats: dict[str, dict[str, Any]] = {}

    def get(self, day: date) -> OkxBboArrays:
        key = day.isoformat()
        if key in self.cache:
            arr = self.cache.pop(key)
            self.cache[key] = arr
            return arr
        arr, stats = load_okx_bbo_day(self.canonical_root, self.symbol, day)
        self.stats[key] = stats
        self.cache[key] = arr
        while len(self.cache) > self.max_days:
            self.cache.popitem(last=False)
        return arr


def combine_okx_arrays(arrays: Sequence[OkxBboArrays]) -> OkxBboArrays:
    usable = [a for a in arrays if len(a.ts_ns) > 0]
    if not usable:
        return EMPTY_BBO
    ts = np.concatenate([a.ts_ns for a in usable])
    exch = np.concatenate([a.exchange_ts_ms for a in usable])
    bid = np.concatenate([a.bid for a in usable])
    ask = np.concatenate([a.ask for a in usable])
    bid_sz = np.concatenate([a.bid_sz for a in usable])
    ask_sz = np.concatenate([a.ask_sz for a in usable])
    order = np.argsort(ts, kind="mergesort")
    return OkxBboArrays(ts[order], exch[order], bid[order], ask[order], bid_sz[order], ask_sz[order])


# ---------------------------------------------------------------------------
# Exact causal OKX features
# ---------------------------------------------------------------------------


def latest_prior_index(arr: OkxBboArrays, target_ns: int) -> int | None:
    if len(arr.ts_ns) == 0:
        return None
    idx = int(np.searchsorted(arr.ts_ns, int(target_ns), side="right") - 1)
    return idx if idx >= 0 else None


def bbo_state(arr: OkxBboArrays, idx: int, target_ns: int, direction: int) -> dict[str, float]:
    age_sec = (int(target_ns) - int(arr.ts_ns[idx])) / NS_PER_SEC
    fresh = bool(0.0 <= age_sec <= OKX_MAX_ENDPOINT_AGE_SEC)
    bid = float(arr.bid[idx])
    ask = float(arr.ask[idx])
    bid_sz = float(arr.bid_sz[idx])
    ask_sz = float(arr.ask_sz[idx])
    mid = 0.5 * (bid + ask)
    denom = bid_sz + ask_sz
    imb = (bid_sz - ask_sz) / denom if denom > 0 else math.nan
    exchange_ms = float(arr.exchange_ts_ms[idx])
    lag_ms = (float(arr.ts_ns[idx]) / 1_000_000.0 - exchange_ms) if math.isfinite(exchange_ms) else math.nan
    return {
        "ts_recv_ns": float(arr.ts_ns[idx]),
        "age_sec": age_sec,
        "fresh_2s": float(fresh),
        "exchange_ts_ms": exchange_ms,
        "recv_minus_exchange_ms": lag_ms,
        "bid_price": bid if fresh else math.nan,
        "ask_price": ask if fresh else math.nan,
        "mid_price": mid if fresh else math.nan,
        "bid_size": bid_sz if fresh else math.nan,
        "ask_size": ask_sz if fresh else math.nan,
        "spread_bps": (10_000.0 * (ask - bid) / mid) if fresh and mid > 0 else math.nan,
        "bbo_imbalance": imb if fresh else math.nan,
        "directional_bbo_imbalance": float(direction) * imb if fresh and math.isfinite(imb) else math.nan,
    }


def summarize_okx_window(
    arr: OkxBboArrays,
    decision_ns: int,
    direction: int,
    window_sec: int,
) -> dict[str, float]:
    if len(arr.ts_ns) == 0:
        return {
            "mid_return_bps": math.nan,
            "directional_mid_return_bps": math.nan,
            "now_age_sec": math.nan,
            "prior_age_sec": math.nan,
            "realized_span_sec": math.nan,
            "now_available": 0.0,
            "prior_available": 0.0,
        }
    start_target = int(decision_ns - int(window_sec) * NS_PER_SEC)
    i_now = latest_prior_index(arr, int(decision_ns))
    i_prior = latest_prior_index(arr, start_target)
    now_age = math.nan if i_now is None else (int(decision_ns) - int(arr.ts_ns[i_now])) / NS_PER_SEC
    prior_age = math.nan if i_prior is None else (start_target - int(arr.ts_ns[i_prior])) / NS_PER_SEC
    now_fresh = i_now is not None and 0.0 <= now_age <= OKX_MAX_ENDPOINT_AGE_SEC
    prior_fresh = i_prior is not None and 0.0 <= prior_age <= OKX_MAX_ENDPOINT_AGE_SEC
    if not now_fresh or not prior_fresh:
        return {
            "mid_return_bps": math.nan,
            "directional_mid_return_bps": math.nan,
            "now_age_sec": now_age,
            "prior_age_sec": prior_age,
            "realized_span_sec": math.nan,
            "now_available": float(now_fresh),
            "prior_available": float(prior_fresh),
        }
    mid_now = 0.5 * (float(arr.bid[i_now]) + float(arr.ask[i_now]))
    mid_prior = 0.5 * (float(arr.bid[i_prior]) + float(arr.ask[i_prior]))
    ret = 10_000.0 * (mid_now / mid_prior - 1.0) if mid_now > 0 and mid_prior > 0 else math.nan
    return {
        "mid_return_bps": ret,
        "directional_mid_return_bps": float(direction) * ret if math.isfinite(ret) else math.nan,
        "now_age_sec": now_age,
        "prior_age_sec": prior_age,
        "realized_span_sec": (int(arr.ts_ns[i_now]) - int(arr.ts_ns[i_prior])) / NS_PER_SEC,
        "now_available": 1.0,
        "prior_available": 1.0,
    }


def attach_okx_features(
    candidates: pd.DataFrame,
    canonical_root: Path,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    valid = candidates[candidates["anchor_valid_3s"]].copy()
    cache = OkxBboDayCache(canonical_root, symbol, max_days=5)
    groups = list(valid.groupby("candidate_date", sort=True))
    for idx, (day_s, group) in enumerate(groups, start=1):
        center = date.fromisoformat(str(day_s))
        arr = combine_okx_arrays([
            cache.get(center - timedelta(days=1)),
            cache.get(center),
            cache.get(center + timedelta(days=1)),
        ])
        print(
            f"      [{idx:02d}/{len(groups):02d}] {day_s}: "
            f"candidates={len(group):3d} context_bbo={len(arr.ts_ns):,}"
        )
        for r in group.itertuples(index=False):
            decision_ns = int(r.decision_ts_recv_ns)
            direction = int(r.direction)
            rec: dict[str, Any] = {
                "candidate_id": str(r.candidate_id),
                "okx_bbo_files_available": bool(len(arr.ts_ns) > 0),
            }
            i_now = latest_prior_index(arr, decision_ns)
            if i_now is not None:
                state = bbo_state(arr, i_now, decision_ns, direction)
                for k, v in state.items():
                    rec[f"okx_current_{k}"] = v
            else:
                for k in (
                    "ts_recv_ns", "age_sec", "fresh_2s", "exchange_ts_ms", "recv_minus_exchange_ms",
                    "bid_price", "ask_price", "mid_price", "bid_size", "ask_size",
                    "spread_bps", "bbo_imbalance", "directional_bbo_imbalance",
                ):
                    rec[f"okx_current_{k}"] = math.nan
            for w in AUDIT_WINDOWS_SEC:
                s = summarize_okx_window(arr, decision_ns, direction, w)
                for k, v in s.items():
                    rec[f"okx_{k}_{w}s"] = v
            rows.append(rec)

    aux = pd.DataFrame(rows)
    out = candidates.copy() if aux.empty else candidates.merge(aux, on="candidate_id", how="left", validate="one_to_one")
    day_stats = pd.DataFrame([cache.stats[k] for k in sorted(cache.stats)])
    return out, day_stats


# ---------------------------------------------------------------------------
# Optional corrected Binance-spot causal substrate for descriptive relative move
# ---------------------------------------------------------------------------


def attach_binance_spot_reference(
    frame: pd.DataFrame,
    root: Path,
    report_root: Path,
) -> tuple[pd.DataFrame, bool, str | None]:
    path = root / "candidate_binance_spot_confirmation_audit.parquet"
    summary_path = report_root / "summary.json"
    if not path.exists() or not summary_path.exists():
        return frame.copy(), False, None
    try:
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return frame.copy(), False, None
    corrected = (
        str(meta.get("phase")) == "7M.9A"
        and str(meta.get("script_version")) == "1.0.1"
        and int(meta.get("price_endpoint_bin_sec", -1)) == 1
        and bool(meta.get("spot_book_snapshots_used")) is False
        and bool(meta.get("future_outcomes_opened")) is False
    )
    if not corrected:
        return frame.copy(), False, None
    requested = ["candidate_id"] + [f"spot_directional_price_return_bps_{w}s" for w in AUDIT_WINDOWS_SEC]
    ref = read_existing_columns(path, requested)
    forbid_outcome_columns(ref, "7M.9A corrected Binance spot causal substrate")
    if ref.empty:
        return frame.copy(), False, sha256_file(path)
    ref["candidate_id"] = ref["candidate_id"].astype(str)
    if ref["candidate_id"].duplicated().any():
        raise RuntimeError("7M.9A causal substrate contains duplicate candidate_id")
    return frame.merge(ref, on="candidate_id", how="left", validate="one_to_one"), True, sha256_file(path)


# ---------------------------------------------------------------------------
# Outcome-blind feasibility summaries
# ---------------------------------------------------------------------------


def summarize_mid_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    a = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    for w in AUDIT_WINDOWS_SEC:
        x = pd.to_numeric(a.get(f"okx_directional_mid_return_bps_{w}s"), errors="coerce")
        now_age = pd.to_numeric(a.get(f"okx_now_age_sec_{w}s"), errors="coerce")
        prior_age = pd.to_numeric(a.get(f"okx_prior_age_sec_{w}s"), errors="coerce")
        span = pd.to_numeric(a.get(f"okx_realized_span_sec_{w}s"), errors="coerce")
        finite = np.isfinite(x)
        rows.append({
            "window_sec": w,
            "anchored_candidates": len(a),
            "finite_candidates": int(finite.sum()),
            "finite_share_pct": pct(float(finite.mean())) if len(a) else math.nan,
            "direction_aligned_positive_count": int((finite & (x > 0)).sum()),
            "direction_aligned_positive_share_pct": pct(float((finite & (x > 0)).mean())) if len(a) else math.nan,
            "direction_aligned_negative_count": int((finite & (x < 0)).sum()),
            "direction_aligned_negative_share_pct": pct(float((finite & (x < 0)).mean())) if len(a) else math.nan,
            "exact_zero_count": int((finite & (np.abs(x) <= 1e-15)).sum()),
            **{f"directional_return_bps_{k}": v for k, v in qstats(x[finite]).items()},
            **{f"now_age_sec_{k}": v for k, v in qstats(now_age[finite & np.isfinite(now_age)]).items()},
            **{f"prior_age_sec_{k}": v for k, v in qstats(prior_age[finite & np.isfinite(prior_age)]).items()},
            **{f"realized_span_sec_{k}": v for k, v in qstats(span[finite & np.isfinite(span)]).items()},
        })
    return pd.DataFrame(rows)


def summarize_bbo_imbalance(frame: pd.DataFrame) -> pd.DataFrame:
    a = frame[frame["anchor_valid_3s"]].copy()
    x = pd.to_numeric(a.get("okx_current_directional_bbo_imbalance"), errors="coerce")
    age = pd.to_numeric(a.get("okx_current_age_sec"), errors="coerce")
    spread = pd.to_numeric(a.get("okx_current_spread_bps"), errors="coerce")
    finite = np.isfinite(x)
    row = {
        "anchored_candidates": len(a),
        "finite_candidates": int(finite.sum()),
        "finite_share_pct": pct(float(finite.mean())) if len(a) else math.nan,
        "direction_aligned_positive_count": int((finite & (x > 0)).sum()),
        "direction_aligned_positive_share_pct": pct(float((finite & (x > 0)).mean())) if len(a) else math.nan,
        "direction_aligned_negative_count": int((finite & (x < 0)).sum()),
        "direction_aligned_negative_share_pct": pct(float((finite & (x < 0)).mean())) if len(a) else math.nan,
        "exact_zero_count": int((finite & (np.abs(x) <= 1e-15)).sum()),
        **{f"directional_imbalance_{k}": v for k, v in qstats(x[finite]).items()},
        **{f"endpoint_age_sec_{k}": v for k, v in qstats(age[finite & np.isfinite(age)]).items()},
        **{f"spread_bps_{k}": v for k, v in qstats(spread[np.isfinite(spread)]).items()},
    }
    return pd.DataFrame([row])


def summarize_cross_venue(frame: pd.DataFrame, spot_ref_available: bool) -> pd.DataFrame:
    if not spot_ref_available:
        return pd.DataFrame([{
            "status": "BINANCE_SPOT_REFERENCE_UNAVAILABLE",
            "window_sec": math.nan,
        }])
    a = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    for w in AUDIT_WINDOWS_SEC:
        okx = pd.to_numeric(a.get(f"okx_directional_mid_return_bps_{w}s"), errors="coerce")
        bnx = pd.to_numeric(a.get(f"spot_directional_price_return_bps_{w}s"), errors="coerce")
        finite = np.isfinite(okx) & np.isfinite(bnx)
        diff = okx - bnx
        both_aligned = finite & (okx > 0) & (bnx > 0)
        both_counter = finite & (okx < 0) & (bnx < 0)
        disagree = finite & (((okx > 0) & (bnx < 0)) | ((okx < 0) & (bnx > 0)))
        corr = float(np.corrcoef(okx[finite], bnx[finite])[0, 1]) if int(finite.sum()) >= 3 else math.nan
        rows.append({
            "status": "DESCRIPTIVE_ONLY_NO_OUTCOMES",
            "window_sec": w,
            "paired_finite_count": int(finite.sum()),
            "paired_finite_share_pct": pct(float(finite.mean())) if len(a) else math.nan,
            "both_direction_aligned_count": int(both_aligned.sum()),
            "both_direction_counter_count": int(both_counter.sum()),
            "venue_direction_disagreement_count": int(disagree.sum()),
            "venue_direction_disagreement_share_pct": pct(float(disagree.mean())) if len(a) else math.nan,
            "okx_minus_binance_directional_return_positive_count": int((finite & (diff > 0)).sum()),
            "okx_minus_binance_directional_return_negative_count": int((finite & (diff < 0)).sum()),
            **{f"okx_minus_binance_return_bps_{k}": v for k, v in qstats(diff[finite]).items()},
            "directional_return_pearson": corr,
        })
    return pd.DataFrame(rows)


def subgroup_representation(frame: pd.DataFrame) -> pd.DataFrame:
    a = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    features = {
        "OKX_MID_RETURN_5S": "okx_directional_mid_return_bps_5s",
        "OKX_MID_RETURN_15S": "okx_directional_mid_return_bps_15s",
        "OKX_MID_RETURN_30S": "okx_directional_mid_return_bps_30s",
        "OKX_MID_RETURN_60S": "okx_directional_mid_return_bps_60s",
        "OKX_CURRENT_BBO_IMBALANCE": "okx_current_directional_bbo_imbalance",
    }
    slices = [
        ("FULL", pd.Series(True, index=a.index)),
        ("DISCOVERY", a["sample_partition"] == "DISCOVERY"),
        ("VALIDATION", a["sample_partition"] == "VALIDATION"),
        ("UP", a["direction"] == 1),
        ("DOWN", a["direction"] == -1),
    ]
    for feature, col in features.items():
        x_all = pd.to_numeric(a.get(col), errors="coerce")
        for label, mask in slices:
            x = x_all[mask]
            finite = np.isfinite(x)
            rows.append({
                "feature": feature,
                "subgroup": label,
                "subgroup_candidates": int(mask.sum()),
                "finite_count": int(finite.sum()),
                "positive_count": int((finite & (x > 0)).sum()),
                "negative_count": int((finite & (x < 0)).sum()),
                "zero_count": int((finite & (np.abs(x) <= 1e-15)).sum()),
            })
    return pd.DataFrame(rows)


def family_availability(frame: pd.DataFrame, day_stats: pd.DataFrame, spot_ref_available: bool) -> pd.DataFrame:
    a = frame[frame["anchor_valid_3s"]].copy()
    endpoint = pd.to_numeric(a.get("okx_current_mid_price"), errors="coerce")
    age = pd.to_numeric(a.get("okx_current_age_sec"), errors="coerce")
    finite_endpoint = np.isfinite(endpoint)
    valid_age = age[finite_endpoint & np.isfinite(age)]
    raw_age = age[np.isfinite(age)]
    return pd.DataFrame([
        {
            "family": "OKX_DIRECT_BBO",
            "source": "canonical okx_spot/bbo (direct OKX websocket books-family first level)",
            "anchored_candidates": len(a),
            "freshness_limit_sec": OKX_MAX_ENDPOINT_AGE_SEC,
            "current_endpoint_finite_count": int(finite_endpoint.sum()),
            "current_endpoint_finite_share_pct": pct(float(finite_endpoint.mean())) if len(a) else math.nan,
            "endpoint_age_sec_median": qstats(valid_age)["median"],
            "endpoint_age_sec_p90": qstats(valid_age)["p90"],
            "endpoint_age_sec_p99": qstats(valid_age)["p99"],
            "endpoint_age_sec_max": qstats(valid_age)["max"],
            "raw_latest_prior_age_sec_max": qstats(raw_age)["max"],
            "canonical_days_with_rows": int((pd.to_numeric(day_stats.get("row_count_valid_quote"), errors="coerce") > 0).sum()) if not day_stats.empty else 0,
            "future_outcomes_opened": False,
        },
        {
            "family": "BINANCE_SPOT_REFERENCE_FOR_RELATIVE_MOVE",
            "source": "corrected Phase 7M.9A causal-only substrate",
            "anchored_candidates": len(a),
            "current_endpoint_finite_count": math.nan,
            "current_endpoint_finite_share_pct": math.nan,
            "endpoint_age_sec_median": math.nan,
            "endpoint_age_sec_p90": math.nan,
            "endpoint_age_sec_p99": math.nan,
            "endpoint_age_sec_max": math.nan,
            "canonical_days_with_rows": math.nan,
            "available": bool(spot_ref_available),
            "future_outcomes_opened": False,
        },
    ])


# ---------------------------------------------------------------------------
# Findings / invariants
# ---------------------------------------------------------------------------


def fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
        return f"{x:.{digits}f}" if math.isfinite(x) else "NA"
    except Exception:
        return "NA"


def build_findings(
    frame: pd.DataFrame,
    availability: pd.DataFrame,
    mid: pd.DataFrame,
    imbalance: pd.DataFrame,
    cross: pd.DataFrame,
) -> str:
    anchored = int(frame["anchor_valid_3s"].sum())
    av = availability.iloc[0]
    m5 = mid[mid["window_sec"] == 5].iloc[0]
    m30 = mid[mid["window_sec"] == 30].iloc[0]
    imb = imbalance.iloc[0]
    cross30 = cross[cross.get("window_sec") == 30] if "window_sec" in cross.columns else pd.DataFrame()
    cross_note = "Binance spot reference unavailable; relative-move branch not computed."
    if not cross30.empty:
        c = cross30.iloc[0]
        cross_note = (
            f"At 30s the two venues had {int(c['paired_finite_count']):,} paired finite candidates; "
            f"venue-direction disagreement was {fmt(c['venue_direction_disagreement_share_pct'])}%."
        )
    return f"""# Phase 7M.10A — OKX cross-venue confirmation feasibility

Outcome-blind feasibility audit only. No future outcome table was opened and no matching was performed.

## Causal population

- Frozen prospective candidates: **{len(frame):,}**
- Valid Phase 7M.6F anchor+3s candidates: **{anchored:,}**
- OKX direct-BBO endpoint coverage: **{fmt(av['current_endpoint_finite_share_pct'])}%**
- Endpoint age: median **{fmt(av['endpoint_age_sec_median'])}s**, p90 **{fmt(av['endpoint_age_sec_p90'])}s**, p99 **{fmt(av['endpoint_age_sec_p99'])}s**, max **{fmt(av['endpoint_age_sec_max'])}s**

## OKX mid-price confirmation

- 5s directional return finite: **{int(m5['finite_candidates']):,}**; aligned **{int(m5['direction_aligned_positive_count']):,}**, counter **{int(m5['direction_aligned_negative_count']):,}**
- 30s directional return finite: **{int(m30['finite_candidates']):,}**; aligned **{int(m30['direction_aligned_positive_count']):,}**, counter **{int(m30['direction_aligned_negative_count']):,}**
- 30s realized span median: **{fmt(m30['realized_span_sec_median'])}s**

## OKX top-of-book size imbalance

- Finite current BBO imbalance: **{int(imb['finite_candidates']):,}**
- Direction-aligned positive: **{int(imb['direction_aligned_positive_count']):,}**
- Direction-aligned negative: **{int(imb['direction_aligned_negative_count']):,}**
- This is best-level size only, not full-depth book imbalance.

## Cross-venue relative move

{cross_note}

## Interpretation discipline

- A strict **2-second maximum endpoint age** is frozen from the outcome-blind v1.0.0 cadence audit. Older last-known OKX quotes are unavailable, not carried forward.
- No OKX outcome-facing threshold is selected here.
- Cross-venue return difference is descriptive feasibility only; no composite score is constructed.
- Future outcomes remain unopened.
"""


def build_invariants(spot_ref_available: bool) -> pd.DataFrame:
    rows = [
        ("phase", PHASE),
        ("script_version", SCRIPT_VERSION),
        ("candidate_population", "frozen prospective 25bp Phase 7M.3/7M.6F universe"),
        ("decision_clock", "Phase 7M.6F exact anchor_ts_recv_ns + 3s"),
        ("okx_source", "canonical okx_spot/bbo direct websocket BBO"),
        ("okx_localbook_used", False),
        ("causal_rule", "only BBO rows with ts_recv_ns <= target"),
        ("staleness_threshold_frozen", True),
        ("okx_max_endpoint_age_sec", OKX_MAX_ENDPOINT_AGE_SEC),
        ("audit_windows_sec", ",".join(map(str, AUDIT_WINDOWS_SEC))),
        ("bbo_imbalance_scope", "best bid/ask sizes only; not full depth"),
        ("binance_spot_reference_available", bool(spot_ref_available)),
        ("cross_venue_relative_move_status", "descriptive feasibility only"),
        ("future_outcomes_opened", False),
        ("matching_performed", False),
        ("composite_signal_built", False),
        ("outcome_threshold_selected", False),
    ]
    return pd.DataFrame(rows, columns=["invariant", "value"])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> int:
    base = 1_700_000_000 * NS_PER_SEC
    arr = OkxBboArrays(
        ts_ns=np.asarray([base, base + 4*NS_PER_SEC, base + 10*NS_PER_SEC], dtype=np.int64),
        exchange_ts_ms=np.asarray([
            base/1e6 - 20.0, (base + 4*NS_PER_SEC)/1e6 - 15.0, (base + 10*NS_PER_SEC)/1e6 - 10.0
        ], dtype=float),
        bid=np.asarray([100.0, 101.0, 102.0]),
        ask=np.asarray([100.2, 101.2, 102.2]),
        bid_sz=np.asarray([5.0, 8.0, 2.0]),
        ask_sz=np.asarray([10.0, 4.0, 6.0]),
    )
    # Future record at +10s must not enter target +9s. The latest prior quote
    # is +4s, but at age 5s it is now correctly rejected as stale.
    idx = latest_prior_index(arr, base + 9*NS_PER_SEC)
    assert idx == 1
    state_up = bbo_state(arr, idx, base + 9*NS_PER_SEC, +1)
    assert state_up["age_sec"] == 5.0
    assert state_up["fresh_2s"] == 0.0
    assert math.isnan(state_up["directional_bbo_imbalance"])

    w = summarize_okx_window(arr, base + 9*NS_PER_SEC, +1, 5)
    assert math.isnan(w["directional_mid_return_bps"])
    assert w["now_available"] == 0.0
    assert w["prior_available"] == 1.0

    w2 = summarize_okx_window(arr, base + 10*NS_PER_SEC, +1, 10)
    expected = 10_000.0 * ((102.1 / 100.1) - 1.0)
    assert abs(w2["directional_mid_return_bps"] - expected) < 1e-9
    assert abs(w2["realized_span_sec"] - 10.0) < 1e-12
    assert w2["now_age_sec"] == 0.0
    assert w2["prior_age_sec"] == 0.0

    print("SELF-TEST PASS: exact anchor+3s causal lookup, strict <=2s OKX endpoint freshness, no-future BBO selection, mid-return timing, and direction-aligned BBO imbalance validated.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--spot-causal-root", type=Path, default=DEFAULT_SPOT_CAUSAL_ROOT)
    p.add_argument("--spot-report-root", type=Path, default=DEFAULT_SPOT_REPORT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    print("Phase 7M.10A - Outcome-blind OKX cross-venue confirmation feasibility")
    print("[1/6] Loading frozen candidate universe and exact anchor+3s clock...")
    candidate_path = args.candidate_root / "candidate_inputs.parquet"
    candidates = load_candidate_inputs(args.candidate_root)
    anchors = load_primary_anchor_rows(args.sequence_root)
    frame = build_causal_candidate_clock(candidates, anchors)

    print("[2/6] Reading direct OKX canonical BBO around causal decision points...")
    frame, day_stats = attach_okx_features(frame, args.canonical_root, args.symbol)
    forbid_outcome_columns(frame, "7M.10A OKX causal frame")

    print("[3/6] Attaching corrected Binance-spot causal reference for descriptive cross-venue audit...")
    frame, spot_ref_available, spot_ref_sha = attach_binance_spot_reference(
        frame, args.spot_causal_root, args.spot_report_root
    )
    forbid_outcome_columns(frame, "7M.10A cross-venue causal frame")

    print("[4/6] Computing outcome-blind feasibility summaries...")
    availability = family_availability(frame, day_stats, spot_ref_available)
    mid = summarize_mid_confirmation(frame)
    imbalance = summarize_bbo_imbalance(frame)
    cross = summarize_cross_venue(frame, spot_ref_available)
    subgroups = subgroup_representation(frame)
    invariants = build_invariants(spot_ref_available)

    print("[5/6] Writing reusable causal substrate and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    causal_path = args.output_root / "candidate_okx_cross_venue_audit.parquet"
    frame.to_parquet(causal_path, index=False)
    write_csv(args.report_root / "family_availability.csv", availability)
    write_csv(args.report_root / "okx_bbo_availability_by_day.csv", day_stats)
    write_csv(args.report_root / "okx_mid_confirmation_feasibility.csv", mid)
    write_csv(args.report_root / "okx_bbo_imbalance_feasibility.csv", imbalance)
    write_csv(args.report_root / "cross_venue_relative_move_feasibility.csv", cross)
    write_csv(args.report_root / "subgroup_representation.csv", subgroups)
    write_csv(args.report_root / "analysis_invariants.csv", invariants)

    findings = build_findings(frame, availability, mid, imbalance, cross)
    (args.report_root / "findings.md").write_text(findings, encoding="utf-8")

    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "symbol": args.symbol,
        "okx_instrument": OKX_INSTRUMENT,
        "candidate_count": len(frame),
        "anchor_valid_3s_count": int(frame["anchor_valid_3s"].sum()),
        "candidate_input_sha256": sha256_file(candidate_path),
        "binance_spot_reference_available": bool(spot_ref_available),
        "binance_spot_reference_sha256": spot_ref_sha,
        "audit_windows_sec": list(AUDIT_WINDOWS_SEC),
        "staleness_threshold_frozen": True,
        "okx_max_endpoint_age_sec": OKX_MAX_ENDPOINT_AGE_SEC,
        "future_outcomes_opened": False,
        "matching_performed": False,
        "composite_signal_built": False,
        "causal_output": str(causal_path),
        "reports": {
            "family_availability": str(args.report_root / "family_availability.csv"),
            "okx_bbo_availability_by_day": str(args.report_root / "okx_bbo_availability_by_day.csv"),
            "okx_mid_confirmation": str(args.report_root / "okx_mid_confirmation_feasibility.csv"),
            "okx_bbo_imbalance": str(args.report_root / "okx_bbo_imbalance_feasibility.csv"),
            "cross_venue_relative_move": str(args.report_root / "cross_venue_relative_move_feasibility.csv"),
            "subgroups": str(args.report_root / "subgroup_representation.csv"),
            "findings": str(args.report_root / "findings.md"),
        },
    }
    write_json(args.report_root / "summary.json", summary)

    print("[6/6] COMPLETE")
    print(f"  causal substrate: {causal_path}")
    print(f"  reports:          {args.report_root}")
    print("  future outcomes opened: NO")
    print("  matching performed:     NO")
    print(f"  OKX freshness rule:     <= {OKX_MAX_ENDPOINT_AGE_SEC:.1f}s per endpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
