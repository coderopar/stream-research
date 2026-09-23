#!/usr/bin/env python3
"""Phase 7M.7A — Outcome-blind non-depth perp-pressure feasibility audit.

Research question
-----------------
After Phase 7M.2–7M.6 largely exhausted the visible order-book/refill branch,
can lightweight perp-only causal streams add genuinely different information
around the same frozen prospective 25-bp candidate population?

This script audits three families WITHOUT opening future outcome labels:

1. Open interest (position creation/contraction context)
2. Observed force-order/liquidation activity (selected/censored observations;
   never interpreted as total liquidation volume)
3. Executed-trade size/composition (concentration conditional on directional
   aggressive quote)

Causal decision point
---------------------
For candidates with a valid Phase 7M.6F anchor, the decision timestamp is:

    exact recovered anchor_ts_recv_ns + 3 seconds

OI/liquidation features are taken only from audited one-second feature rows
whose ``available_after_second`` is <= the decision second. Raw trades are
included only when ``ts_recv_ns <= decision_ts_recv_ns``.

No outcomes, no matching, no thresholds selected from outcomes, no composite
signal. The output is a feasibility/audit substrate for a later Phase 7M.7B
hypothesis freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0.0"
PHASE = "7M.7A"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
AUDIT_WINDOWS_SEC = (5, 15, 30, 60)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_exact_participant_interaction_v1"
)
DEFAULT_SEQUENCE_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_refill_redepletion_sequence_v2"
)
DEFAULT_FLOW_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "flow_features_1s_v1"
DEFAULT_CORE_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "core_1s_v1"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_non_depth_pressure_feasibility_v1"
)
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_non_depth_pressure_feasibility"

FLOW_CAUSAL_COLUMNS = [
    "receive_second",
    "available_after_second",
    "flow_feature_context_ready",
    "open_interest",
]
for _w in AUDIT_WINDOWS_SEC:
    FLOW_CAUSAL_COLUMNS += [
        f"open_interest_change_{_w}s",
        f"open_interest_change_{_w}s_pct",
        f"liquidation_notional_{_w}s",
        f"net_liquidation_notional_{_w}s",
        f"liquidation_imbalance_{_w}s",
        f"liquidation_to_quote_ratio_{_w}s",
        f"trade_count_{_w}s",
        f"quote_volume_{_w}s",
    ]

CORE_CAUSAL_COLUMNS = [
    "receive_second",
    "open_interest",
    "open_interest_updated",
    "open_interest_age_seconds",
    "open_interest_fresh",
    "liquidation_event_count",
    "long_liquidation_notional",
    "short_liquidation_notional",
    "liquidation_notional",
    "net_liquidation_notional",
]

RAW_TRADE_COLUMNS = [
    "ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker"
]

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


def safe_bool(x: Any) -> bool:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    try:
        return bool(x)
    except Exception:
        return False


def pct(x: float) -> float:
    return float(x) * 100.0


def qstats(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {k: math.nan for k in ("min", "p10", "p25", "median", "p75", "p90", "p99", "max")}
    qs = np.quantile(a, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "p99", "max"), map(float, qs)))


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def partition_path(root: Path, symbol: str, day: date) -> Path:
    return root / f"symbol={symbol}" / f"date={day.isoformat()}" / f"part-{day.isoformat()}.parquet"


def read_existing_columns(path: Path, requested: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=requested)
    schema = pq.ParquetFile(path).schema_arrow
    names = set(schema.names)
    use = [c for c in requested if c in names]
    if not use:
        return pd.DataFrame(columns=requested)
    out = pq.read_table(path, columns=use).to_pandas()
    for c in requested:
        if c not in out.columns:
            out[c] = np.nan
    return out[requested]


def forbid_outcome_columns(frame: pd.DataFrame, context: str) -> None:
    bad = []
    for c in frame.columns:
        cl = str(c).lower()
        if any(tok in cl for tok in FORBIDDEN_OUTCOME_TOKENS):
            bad.append(c)
    if bad:
        raise RuntimeError(f"{context}: future/outcome-like columns are forbidden in Phase 7M.7A: {bad}")


# ---------------------------------------------------------------------------
# Frozen candidate + anchor loading
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
    parts: list[pd.DataFrame] = []
    cols = [
        "candidate_id", "candidate_date", "observation_horizon_sec",
        "anchor_found", "anchor_ts_recv_ns", "horizon_complete", "invalid_reason",
    ]
    for p in paths:
        f = read_existing_columns(p, cols)
        if f.empty:
            continue
        h = pd.to_numeric(f["observation_horizon_sec"], errors="coerce")
        f = f[h == PRIMARY_HORIZON_SEC].copy()
        if not f.empty:
            parts.append(f)
    if not parts:
        raise RuntimeError("No 3-second Phase 7M.6F candidate metric rows found")
    a = pd.concat(parts, ignore_index=True)
    if a["candidate_id"].duplicated().any():
        dup = a.loc[a["candidate_id"].duplicated(), "candidate_id"].head().tolist()
        raise RuntimeError(f"Duplicate primary anchor rows: {dup}")
    a["anchor_ts_recv_ns"] = pd.to_numeric(a["anchor_ts_recv_ns"], errors="coerce")
    return a


def build_causal_candidate_clock(candidates: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["candidate_id", "anchor_found", "anchor_ts_recv_ns", "horizon_complete", "invalid_reason"] if c in anchors.columns]
    out = candidates.merge(anchors[cols], on="candidate_id", how="left", validate="one_to_one")
    anchor_ok = out.get("anchor_found", False)
    if not isinstance(anchor_ok, pd.Series):
        anchor_ok = pd.Series(False, index=out.index)
    anchor_ok = anchor_ok.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce"))
    out["anchor_valid_3s"] = anchor_ok
    out["decision_ts_recv_ns"] = np.where(
        anchor_ok,
        pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce") + PRIMARY_HORIZON_SEC * 1_000_000_000,
        np.nan,
    )
    # A one-second feature row is safe only when its available_after_second is
    # no later than this whole-second cutoff. Raw trades use exact ns below.
    out["decision_second_floor"] = np.floor(pd.to_numeric(out["decision_ts_recv_ns"], errors="coerce") / 1e9)
    return out


# ---------------------------------------------------------------------------
# Audited one-second OI/liquidation feature lookup
# ---------------------------------------------------------------------------


def load_day_context(root: Path, symbol: str, center: date, columns: list[str]) -> pd.DataFrame:
    parts = []
    for d in (center - timedelta(days=1), center, center + timedelta(days=1)):
        p = partition_path(root, symbol, d)
        f = read_existing_columns(p, columns)
        if not f.empty:
            parts.append(f)
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True)
    if "receive_second" in out.columns:
        out["receive_second"] = pd.to_numeric(out["receive_second"], errors="coerce")
        out = out[np.isfinite(out["receive_second"])].copy()
        out["receive_second"] = out["receive_second"].astype("int64")
        out = out.sort_values("receive_second", kind="mergesort").drop_duplicates("receive_second", keep="last")
    return out.reset_index(drop=True)


def attach_flow_and_core(candidates: pd.DataFrame, flow_root: Path, core_root: Path, symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = candidates[candidates["anchor_valid_3s"]].copy()
    for day_s, group in valid.groupby("candidate_date", sort=True):
        center = date.fromisoformat(str(day_s))
        flow = load_day_context(flow_root, symbol, center, FLOW_CAUSAL_COLUMNS)
        core = load_day_context(core_root, symbol, center, CORE_CAUSAL_COLUMNS)
        if flow.empty:
            for r in group.itertuples(index=False):
                rows.append({"candidate_id": r.candidate_id, "flow_row_found": False})
            continue

        flow_avail = pd.to_numeric(flow["available_after_second"], errors="coerce").to_numpy(float)
        order = np.argsort(np.nan_to_num(flow_avail, nan=np.inf), kind="mergesort")
        flow = flow.iloc[order].reset_index(drop=True)
        flow_avail = pd.to_numeric(flow["available_after_second"], errors="coerce").to_numpy(float)

        core_map = {}
        if not core.empty:
            core_map = {int(r.receive_second): r for r in core.itertuples(index=False)}

        for r in group.itertuples(index=False):
            cutoff = int(r.decision_second_floor)
            idx = int(np.searchsorted(flow_avail, cutoff, side="right") - 1)
            rec: dict[str, Any] = {"candidate_id": r.candidate_id, "flow_row_found": idx >= 0}
            if idx < 0:
                rows.append(rec)
                continue
            fr = flow.iloc[idx]
            recv_sec = int(fr["receive_second"])
            rec.update({
                "feature_receive_second": recv_sec,
                "feature_available_after_second": safe_float(fr["available_after_second"]),
                "feature_lag_from_decision_sec": float(r.decision_ts_recv_ns / 1e9 - recv_sec),
                "flow_feature_context_ready": safe_bool(fr.get("flow_feature_context_ready", False)),
                "open_interest": safe_float(fr.get("open_interest")),
            })
            for w in AUDIT_WINDOWS_SEC:
                for base in (
                    "open_interest_change", "open_interest_change_{w}s_pct",
                    "liquidation_notional", "net_liquidation_notional",
                    "liquidation_imbalance", "liquidation_to_quote_ratio",
                    "trade_count", "quote_volume",
                ):
                    if base == "open_interest_change_{w}s_pct":
                        c = f"open_interest_change_{w}s_pct"
                    else:
                        c = f"{base}_{w}s"
                    rec[c] = safe_float(fr.get(c))
                net_liq = rec.get(f"net_liquidation_notional_{w}s", math.nan)
                rec[f"directional_observed_liquidation_notional_{w}s"] = (
                    float(r.direction) * net_liq if math.isfinite(net_liq) else math.nan
                )
            cr = core_map.get(recv_sec)
            if cr is not None:
                rec.update({
                    "open_interest_age_seconds": safe_float(getattr(cr, "open_interest_age_seconds", math.nan)),
                    "open_interest_fresh": safe_bool(getattr(cr, "open_interest_fresh", False)),
                    "open_interest_updated_1s": safe_bool(getattr(cr, "open_interest_updated", False)),
                    "observed_liquidation_event_count_1s": safe_float(getattr(cr, "liquidation_event_count", math.nan)),
                    "observed_long_liquidation_notional_1s": safe_float(getattr(cr, "long_liquidation_notional", math.nan)),
                    "observed_short_liquidation_notional_1s": safe_float(getattr(cr, "short_liquidation_notional", math.nan)),
                })
            rows.append(rec)
    aux = pd.DataFrame(rows)
    if aux.empty:
        return candidates.copy()
    return candidates.merge(aux, on="candidate_id", how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# Exact causal trade composition
# ---------------------------------------------------------------------------


def trade_files(canonical_root: Path, day: date) -> list[Path]:
    base = canonical_root / "perps" / "trades" / f"date={day.isoformat()}"
    if base.exists():
        return sorted(base.rglob("*.parquet"))
    # Conservative fallback for alternative partition nesting.
    family = canonical_root / "perps" / "trades"
    if not family.exists():
        return []
    needle = f"date={day.isoformat()}"
    return sorted(p for p in family.rglob("*.parquet") if needle in p.as_posix())


@dataclass
class TradeArrays:
    ts_ns: np.ndarray
    quote: np.ndarray
    is_buy: np.ndarray
    known_side: np.ndarray


def load_trade_arrays(canonical_root: Path, symbol: str, days: list[date]) -> TradeArrays:
    ts_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    buy_parts: list[np.ndarray] = []
    known_parts: list[np.ndarray] = []
    for d in days:
        for path in trade_files(canonical_root, d):
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            use = [c for c in RAW_TRADE_COLUMNS if c in names]
            missing = {"ts_recv_ns", "price", "quantity", "buyer_is_maker"} - set(use)
            if missing:
                raise RuntimeError(f"Trade file {path} missing required columns {sorted(missing)}")
            for batch in pf.iter_batches(batch_size=262144, columns=use):
                f = batch.to_pandas()
                if "symbol" in f.columns:
                    f = f[f["symbol"].astype(str) == symbol]
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
                quote = (pd.to_numeric(f["price"], errors="raise").to_numpy(float) *
                         pd.to_numeric(f["quantity"], errors="raise").to_numpy(float))
                maker = f["buyer_is_maker"].astype("boolean")
                known = maker.notna().to_numpy(bool)
                is_buy = maker.eq(False).fillna(False).to_numpy(bool)
                ts_parts.append(ts_a)
                q_parts.append(quote)
                buy_parts.append(is_buy)
                known_parts.append(known)
    if not ts_parts:
        return TradeArrays(
            np.asarray([], dtype=np.int64), np.asarray([], dtype=float),
            np.asarray([], dtype=bool), np.asarray([], dtype=bool),
        )
    ts = np.concatenate(ts_parts)
    q = np.concatenate(q_parts)
    b = np.concatenate(buy_parts)
    k = np.concatenate(known_parts)
    order = np.argsort(ts, kind="mergesort")
    return TradeArrays(ts[order], q[order], b[order], k[order])


def summarize_trade_window(arr: TradeArrays, decision_ns: int, direction: int, window_sec: int) -> dict[str, float]:
    start_ns = int(decision_ns - window_sec * 1_000_000_000)
    lo = int(np.searchsorted(arr.ts_ns, start_ns, side="right"))
    hi = int(np.searchsorted(arr.ts_ns, int(decision_ns), side="right"))
    if hi <= lo:
        return {
            "trade_count": 0.0, "trade_quote": 0.0, "largest_trade_share": math.nan,
            "trade_size_hhi": math.nan, "directional_trade_count": 0.0,
            "directional_trade_quote": 0.0, "directional_average_trade_quote": math.nan,
            "directional_max_trade_quote": 0.0, "directional_largest_trade_share": math.nan,
            "directional_trade_size_hhi": math.nan, "unknown_aggressor_count": 0.0,
        }
    q = arr.quote[lo:hi]
    known = arr.known_side[lo:hi]
    buy = arr.is_buy[lo:hi]
    total = float(np.sum(q))
    total_sq = float(np.sum(q * q))
    max_q = float(np.max(q)) if len(q) else 0.0
    directional_mask = known & (buy if int(direction) > 0 else ~buy)
    dq = q[directional_mask]
    dtotal = float(np.sum(dq))
    dsq = float(np.sum(dq * dq))
    dmax = float(np.max(dq)) if len(dq) else 0.0
    return {
        "trade_count": float(len(q)),
        "trade_quote": total,
        "largest_trade_share": (max_q / total if total > 0 else math.nan),
        "trade_size_hhi": (total_sq / (total * total) if total > 0 else math.nan),
        "directional_trade_count": float(len(dq)),
        "directional_trade_quote": dtotal,
        "directional_average_trade_quote": (dtotal / len(dq) if len(dq) else math.nan),
        "directional_max_trade_quote": dmax,
        "directional_largest_trade_share": (dmax / dtotal if dtotal > 0 else math.nan),
        "directional_trade_size_hhi": (dsq / (dtotal * dtotal) if dtotal > 0 else math.nan),
        "unknown_aggressor_count": float(np.sum(~known)),
    }


def attach_trade_composition(candidates: pd.DataFrame, canonical_root: Path, symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = candidates[candidates["anchor_valid_3s"]].copy()
    for day_s, group in valid.groupby("candidate_date", sort=True):
        center = date.fromisoformat(str(day_s))
        # 60-second trailing windows can cross midnight, and a +3s decision can
        # fall into the next UTC date for an end-of-day candidate.
        arr = load_trade_arrays(canonical_root, symbol, [center - timedelta(days=1), center, center + timedelta(days=1)])
        for r in group.itertuples(index=False):
            rec: dict[str, Any] = {"candidate_id": r.candidate_id}
            if len(arr.ts_ns) == 0:
                rec["trade_files_available"] = False
                rows.append(rec)
                continue
            rec["trade_files_available"] = True
            decision_ns = int(r.decision_ts_recv_ns)
            for w in AUDIT_WINDOWS_SEC:
                s = summarize_trade_window(arr, decision_ns, int(r.direction), w)
                for k, v in s.items():
                    rec[f"trade_{k}_{w}s"] = v
            rows.append(rec)
    aux = pd.DataFrame(rows)
    if aux.empty:
        return candidates.copy()
    return candidates.merge(aux, on="candidate_id", how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# Feasibility summaries
# ---------------------------------------------------------------------------


def summarize_oi(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(frame)
    anchored = frame[frame["anchor_valid_3s"]].copy()
    age = pd.to_numeric(anchored.get("open_interest_age_seconds"), errors="coerce")
    for w in AUDIT_WINDOWS_SEC:
        c = f"open_interest_change_{w}s_pct"
        vals = pd.to_numeric(anchored.get(c), errors="coerce")
        finite = np.isfinite(vals)
        nonzero = finite & (np.abs(vals) > 1e-15)
        pos = finite & (vals > 0)
        neg = finite & (vals < 0)
        stats = qstats(vals[finite])
        rows.append({
            "window_sec": w,
            "total_candidates": n,
            "anchored_candidates": len(anchored),
            "finite_candidates": int(finite.sum()),
            "finite_share_pct": pct(float(finite.mean())) if len(anchored) else math.nan,
            "nonzero_share_pct": pct(float(nonzero.mean())) if len(anchored) else math.nan,
            "positive_share_pct": pct(float(pos.mean())) if len(anchored) else math.nan,
            "negative_share_pct": pct(float(neg.mean())) if len(anchored) else math.nan,
            "median_oi_age_sec": safe_float(np.nanmedian(age.to_numpy(float))) if np.isfinite(age).any() else math.nan,
            "p90_oi_age_sec": safe_float(np.nanquantile(age.to_numpy(float), 0.90)) if np.isfinite(age).any() else math.nan,
            **{f"oi_change_pct_{k}": v for k, v in stats.items()},
        })
    return pd.DataFrame(rows)


def summarize_liquidations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchored = frame[frame["anchor_valid_3s"]].copy()
    for w in AUDIT_WINDOWS_SEC:
        gross = pd.to_numeric(anchored.get(f"liquidation_notional_{w}s"), errors="coerce")
        directional = pd.to_numeric(anchored.get(f"directional_observed_liquidation_notional_{w}s"), errors="coerce")
        finite = np.isfinite(gross)
        observed = finite & (gross > 0)
        dpos = np.isfinite(directional) & (directional > 0)
        dneg = np.isfinite(directional) & (directional < 0)
        rows.append({
            "window_sec": w,
            "anchored_candidates": len(anchored),
            "finite_candidates": int(finite.sum()),
            "finite_share_pct": pct(float(finite.mean())) if len(anchored) else math.nan,
            "any_observed_force_order_share_pct": pct(float(observed.mean())) if len(anchored) else math.nan,
            "any_observed_force_order_count": int(observed.sum()),
            "same_direction_forced_pressure_share_pct": pct(float(dpos.mean())) if len(anchored) else math.nan,
            "counter_direction_forced_pressure_share_pct": pct(float(dneg.mean())) if len(anchored) else math.nan,
            **{f"gross_notional_{k}": v for k, v in qstats(gross[observed]).items()},
            **{f"directional_net_notional_{k}": v for k, v in qstats(directional[np.isfinite(directional)]).items()},
        })
    return pd.DataFrame(rows)


def summarize_trades(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchored = frame[frame["anchor_valid_3s"]].copy()
    for w in AUDIT_WINDOWS_SEC:
        dq = pd.to_numeric(anchored.get(f"trade_directional_trade_quote_{w}s"), errors="coerce")
        dn = pd.to_numeric(anchored.get(f"trade_directional_trade_count_{w}s"), errors="coerce")
        hhi = pd.to_numeric(anchored.get(f"trade_directional_trade_size_hhi_{w}s"), errors="coerce")
        lshare = pd.to_numeric(anchored.get(f"trade_directional_largest_trade_share_{w}s"), errors="coerce")
        avg = pd.to_numeric(anchored.get(f"trade_directional_average_trade_quote_{w}s"), errors="coerce")
        finite = np.isfinite(dq)
        positive = finite & (dq > 0)
        enough5 = np.isfinite(dn) & (dn >= 5)
        rows.append({
            "window_sec": w,
            "anchored_candidates": len(anchored),
            "trade_file_coverage_pct": pct(float(anchored.get("trade_files_available", False).fillna(False).astype(bool).mean())) if len(anchored) else math.nan,
            "directional_quote_finite_share_pct": pct(float(finite.mean())) if len(anchored) else math.nan,
            "directional_quote_positive_share_pct": pct(float(positive.mean())) if len(anchored) else math.nan,
            "directional_trade_count_ge5_share_pct": pct(float(enough5.mean())) if len(anchored) else math.nan,
            **{f"directional_trade_count_{k}": v for k, v in qstats(dn[np.isfinite(dn)]).items()},
            **{f"directional_quote_{k}": v for k, v in qstats(dq[positive]).items()},
            **{f"directional_average_trade_quote_{k}": v for k, v in qstats(avg[np.isfinite(avg)]).items()},
            **{f"directional_largest_trade_share_{k}": v for k, v in qstats(lshare[np.isfinite(lshare)]).items()},
            **{f"directional_trade_size_hhi_{k}": v for k, v in qstats(hhi[np.isfinite(hhi)]).items()},
        })
    return pd.DataFrame(rows)


def candidate_availability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    anchored = frame[frame["anchor_valid_3s"]].copy()
    rows = [
        {"family": "ANCHOR_3S", "usable": int(frame["anchor_valid_3s"].sum()), "total": len(frame)},
        {"family": "OI_30S", "usable": int(np.isfinite(pd.to_numeric(anchored.get("open_interest_change_30s_pct"), errors="coerce")).sum()), "total": len(anchored)},
        {"family": "LIQUIDATION_30S", "usable": int(np.isfinite(pd.to_numeric(anchored.get("liquidation_notional_30s"), errors="coerce")).sum()), "total": len(anchored)},
        {"family": "TRADE_COMPOSITION_30S", "usable": int(np.isfinite(pd.to_numeric(anchored.get("trade_directional_trade_size_hhi_30s"), errors="coerce")).sum()), "total": len(anchored)},
    ]
    for r in rows:
        r["usable_share_pct"] = pct(r["usable"] / r["total"]) if r["total"] else math.nan
    return pd.DataFrame(rows)


def make_findings(frame: pd.DataFrame, oi: pd.DataFrame, liq: pd.DataFrame, trade: pd.DataFrame) -> str:
    anchored = int(frame["anchor_valid_3s"].sum())
    oi30 = oi.loc[oi["window_sec"] == 30].iloc[0].to_dict()
    liq30 = liq.loc[liq["window_sec"] == 30].iloc[0].to_dict()
    tr30 = trade.loc[trade["window_sec"] == 30].iloc[0].to_dict()
    return f"""# Phase 7M.7A — non-depth perp-pressure feasibility

Status: **OUTCOME-BLIND FEASIBILITY AUDIT COMPLETE**

## Frozen scope

- Prospective 25-bp candidates: **{len(frame):,}**
- Exact 7M.6F anchor +3s decision points available: **{anchored:,}** ({anchored/len(frame)*100:.2f}%)
- Future outcome tables opened: **NO**
- Matching performed: **NO**
- Composite signal built: **NO**

## 30-second causal-window snapshot

### Open interest

- Finite OI change coverage: **{oi30['finite_share_pct']:.2f}%**
- Non-zero OI change share: **{oi30['nonzero_share_pct']:.2f}%**
- Positive OI change share: **{oi30['positive_share_pct']:.2f}%**
- Negative OI change share: **{oi30['negative_share_pct']:.2f}%**
- Median OI state age at the causal feature row: **{oi30['median_oi_age_sec']:.2f}s**
- P90 OI state age: **{oi30['p90_oi_age_sec']:.2f}s**

Interpretation constraint: aggregate OI identifies position expansion/contraction, not whether new positions are longs or shorts.

### Observed force-order / liquidation stream

- Finite rolling coverage: **{liq30['finite_share_pct']:.2f}%**
- Candidates with any observed force-order notional: **{liq30['any_observed_force_order_count']:,}** ({liq30['any_observed_force_order_share_pct']:.2f}%)
- Same-direction forced-pressure sign share: **{liq30['same_direction_forced_pressure_share_pct']:.2f}%**

Interpretation constraint: collector forceOrder observations are selected/censored observations, **not total liquidation volume**.

### Trade-size composition

- Raw-trade file coverage: **{tr30['trade_file_coverage_pct']:.2f}%**
- Positive directional aggressive quote share: **{tr30['directional_quote_positive_share_pct']:.2f}%**
- Candidates with at least 5 directional aggressive trades: **{tr30['directional_trade_count_ge5_share_pct']:.2f}%**
- Median directional trade-size HHI: **{tr30['directional_trade_size_hhi_median']:.6f}**
- Median largest directional trade share: **{tr30['directional_largest_trade_share_median']:.6f}**

## What this phase can decide

This phase determines which of OI, observed force-order activity, and trade-size composition have enough causal coverage and variation to justify a separately frozen 7M.7B hypothesis/matching design.

It does **not** say whether any of them predict P75 continuation.
"""


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def self_test() -> None:
    # Exact causal trade cutoff and direction semantics.
    arr = TradeArrays(
        ts_ns=np.asarray([1, 2, 3, 4, 5], dtype=np.int64) * 1_000_000_000,
        quote=np.asarray([100.0, 200.0, 300.0, 400.0, 500.0]),
        is_buy=np.asarray([True, False, True, False, True]),
        known_side=np.asarray([True, True, True, True, True]),
    )
    up = summarize_trade_window(arr, 4_500_000_000, +1, 5)
    # Includes timestamps 1..4 only; UP directional = buys at 1 and 3.
    assert up["trade_count"] == 4
    assert abs(up["directional_trade_quote"] - 400.0) < 1e-12
    assert up["directional_trade_count"] == 2
    assert abs(up["directional_max_trade_quote"] - 300.0) < 1e-12
    assert abs(up["directional_largest_trade_share"] - 0.75) < 1e-12
    assert abs(up["directional_trade_size_hhi"] - ((100.0**2 + 300.0**2) / 400.0**2)) < 1e-12

    down = summarize_trade_window(arr, 4_500_000_000, -1, 5)
    assert abs(down["directional_trade_quote"] - 600.0) < 1e-12
    assert down["directional_trade_count"] == 2

    # Decision point must be exact anchor+3s.
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
    assert int(cc.loc[0, "decision_second_floor"]) == 103

    print("SELF-TEST PASS: exact anchor+3s clock, causal trade cutoff, and aggressor direction semantics validated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    p.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    p.add_argument("--core-root", type=Path, default=DEFAULT_CORE_ROOT)
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

    print(f"Phase {PHASE} - Outcome-blind OI / liquidation / trade-composition feasibility")
    print("[1/6] Loading frozen 7M.3 causal candidate universe and 7M.6F anchor clock...")
    candidates = load_candidate_inputs(args.candidate_root)
    anchors = load_primary_anchor_rows(args.sequence_root)
    frame = build_causal_candidate_clock(candidates, anchors)
    print(f"      candidates={len(frame):,} anchor+3s={int(frame['anchor_valid_3s'].sum()):,}")

    print("[2/6] Attaching audited causal one-second OI/liquidation context...")
    frame = attach_flow_and_core(frame, args.flow_root, args.core_root, args.symbol)

    print("[3/6] Aggregating exact causal raw-trade composition around candidate decisions...")
    frame = attach_trade_composition(frame, args.canonical_root, args.symbol)

    print("[4/6] Computing outcome-blind family feasibility summaries...")
    oi = summarize_oi(frame)
    liq = summarize_liquidations(frame)
    trade = summarize_trades(frame)
    avail = candidate_availability_summary(frame)

    print("[5/6] Writing causal audit substrate and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    # Explicitly verify the output table contains no future labels before write.
    forbid_outcome_columns(frame, "7M.7A candidate audit output")
    causal_path = args.output_root / "candidate_non_depth_pressure_audit.parquet"
    frame.to_parquet(causal_path, index=False)

    write_csv(args.report_root / "family_availability.csv", avail)
    write_csv(args.report_root / "open_interest_feasibility.csv", oi)
    write_csv(args.report_root / "observed_liquidation_feasibility.csv", liq)
    write_csv(args.report_root / "trade_composition_feasibility.csv", trade)

    invariants = pd.DataFrame([
        {"invariant": "future_outcomes_opened", "value": False},
        {"invariant": "matching_performed", "value": False},
        {"invariant": "composite_signal_built", "value": False},
        {"invariant": "decision_clock", "value": "exact_7m6f_anchor_ts_recv_ns_plus_3s"},
        {"invariant": "flow_feature_causality", "value": "available_after_second<=decision_second_floor"},
        {"invariant": "raw_trade_causality", "value": "ts_recv_ns<=decision_ts_recv_ns"},
        {"invariant": "liquidation_semantics", "value": "selected_censored_forceOrder_observations_not_total_liquidation_volume"},
        {"invariant": "oi_semantics", "value": "aggregate_position_expansion_contraction_not_long_short_identity"},
        {"invariant": "candidate_count", "value": len(frame)},
        {"invariant": "anchor_valid_3s_count", "value": int(frame["anchor_valid_3s"].sum())},
    ])
    write_csv(args.report_root / "analysis_invariants.csv", invariants)

    findings = make_findings(frame, oi, liq, trade)
    (args.report_root / "findings.md").write_text(findings, encoding="utf-8")

    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "symbol": args.symbol,
        "candidate_count": len(frame),
        "anchor_valid_3s_count": int(frame["anchor_valid_3s"].sum()),
        "windows_sec": list(AUDIT_WINDOWS_SEC),
        "future_outcomes_opened": False,
        "matching_performed": False,
        "composite_signal_built": False,
        "candidate_input_sha256": sha256_file(args.candidate_root / "candidate_inputs.parquet"),
        "causal_output": str(causal_path),
        "reports": {
            "family_availability": str(args.report_root / "family_availability.csv"),
            "open_interest": str(args.report_root / "open_interest_feasibility.csv"),
            "observed_liquidations": str(args.report_root / "observed_liquidation_feasibility.csv"),
            "trade_composition": str(args.report_root / "trade_composition_feasibility.csv"),
            "findings": str(args.report_root / "findings.md"),
        },
    }
    write_json(args.report_root / "summary.json", summary)

    print("[6/6] Complete.")
    print(f"      causal={causal_path}")
    print(f"      report={args.report_root}")
    print("      outcomes_opened=NO matching=NO composite=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
