#!/usr/bin/env python3
"""Phase 7M.12A - outcome-blind BTC-options gamma-landscape feasibility.

Research question
-----------------
Immediately BEFORE a prospective early-25bp Binance-perp candidate begins,
how much OBSERVED option hedge sensitivity is concentrated near spot, where is
that sensitivity located by strike / expiry, and is the substrate complete and
variable enough to justify a later matched continuation study?

Primary physical primitive
--------------------------
For each option contract for which BOTH contemporaneous mark/Greek data and a
recent OI observation are available:

    observed_gamma_sensitivity_native = open_interest * abs(gamma)

This is an UNSIGNED sensitivity magnitude. It is NOT dealer gamma, net gamma,
or a prediction of hedge direction.

For cross-price interpretability the audit also records a first-order 1% spot
move proxy:

    observed_delta_change_for_1pct_move
        = open_interest * abs(gamma) * spot_index_price * 0.01

This approximates the magnitude of delta change associated with a 1% underlying
move, in the native option-contract/delta convention supplied by the exchange.
It is still NOT an estimate of actual hedge flow because participant inventory
sign and hedge policy are unknown.

Outcome-blind geometry audited
------------------------------
* total observed OI x |gamma| across the tracked mark/Greek universe;
* sensitivity within +/-1%, +/-2%, and +/-5% of spot;
* fraction of observed sensitivity near spot;
* front-expiry share and expiry concentration (HHI);
* strike concentration (HHI), dominant strike share, and distance from spot;
* sensitivity-weighted strike distance and expiry distance;
* overlap between the broader expiry-scoped OI universe and contracts for
  which contemporaneous gamma was actually collected.

Scientific safeguards
---------------------
* Causal cutoff is the START of candidate_second. No candidate-second option
  data and no future label/outcome file is opened.
* Exact latest mark surface at or before the causal cutoff is resolved from
  canonical integer ts_recv_ns values (no persisted-float timestamp reuse).
* OI uses the latest observation per contract in a broad preceding 30-minute
  FEASIBILITY lookback. That 30-minute lookback is an audit ceiling, NOT a
  frozen scientific freshness threshold for a later exposure study.
* +/-1/2/5% bands are descriptive geometry, not outcome-tuned cutoffs.
* No long/short/dealer sign is inferred from public OI.
* No option-trade side/maker interpretation is used.
* No IV outcome result is used to define or threshold gamma exposure.
* Per-day checkpoints preserve completed work and are contract-versioned.

Interpretation boundaries
-------------------------
A high value means: "there is a large amount of OBSERVED option delta
sensitivity that could change when spot moves." It does NOT mean hedgers must
buy or sell BTC, and it does NOT establish that dealers are net short gamma.
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
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

SCRIPT_VERSION = "1.0.0"
CHECKPOINT_CONTRACT_VERSION = "1.0.0"
PHASE = "7M.12A"
EXPECTED_7M11A_HELPER_VERSION = "1.0.1"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
NS_PER_SEC = 1_000_000_000
OI_AUDIT_LOOKBACK_SEC = 1800  # broad feasibility ceiling, not later science threshold
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"
NEAR_BANDS_PCT = (1.0, 2.0, 5.0)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_exact_participant_interaction_v1"
)
DEFAULT_SEQUENCE_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_refill_redepletion_sequence_v2"
)
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_7M11A_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07m11a_audit_options_regime_anticipation_feasibility.py"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_options_gamma_landscape_feasibility_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_options_gamma_landscape_feasibility"
)

MARK_COLUMNS = (
    "ts_recv_ns", "symbol", "exchange_event_ms", "index_price", "mark_iv",
    "delta", "gamma", "vega",
)
OI_COLUMNS = (
    "ts_recv_ns", "symbol", "exchange_event_ms", "open_interest", "open_interest_value",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "outcome", "future", "p75", "p50", "p100", "mfe", "mae", "success",
    "first_passage", "reach_", "reaches_", "h300_", "h600_", "h900_", "h1800_",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.12A requires pandas, numpy and pyarrow in the project .venv.")


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
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def safe_div(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)) or den == 0:
        return math.nan
    return float(num / den)


def split_period(day_text: str) -> str:
    if str(day_text) <= DISCOVERY_END_DATE:
        return "DISCOVERY"
    if str(day_text) >= VALIDATION_START_DATE:
        return "VALIDATION"
    return "GAP"


def forbid_outcome_columns(frame: pd.DataFrame, label: str) -> None:
    bad = []
    for col in frame.columns:
        lc = str(col).lower()
        if any(tok in lc for tok in FORBIDDEN_OUTCOME_TOKENS):
            bad.append(str(col))
    if bad:
        raise RuntimeError(f"{label} unexpectedly exposes future/outcome columns: {sorted(bad)}")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return math.nan
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    c = np.cumsum(w)
    target = float(q) * float(c[-1])
    idx = int(np.searchsorted(c, target, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def hhi(weights: Iterable[float]) -> float:
    arr = np.asarray(list(weights), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return math.nan
    s = float(arr.sum())
    if s <= 0:
        return math.nan
    p = arr / s
    return float(np.sum(p * p))


def qstats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    keys = ("min", "p10", "p25", "median", "p75", "p90", "p99", "max")
    if len(arr) == 0:
        return {k: math.nan for k in keys}
    qs = np.quantile(arr, [0, .10, .25, .50, .75, .90, .99, 1])
    return dict(zip(keys, map(float, qs)))


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str, allow_nan=True)
        f.write("\n")


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Reuse audited 7M.11A causal readers
# ---------------------------------------------------------------------------


def load_7m11a_helper(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required 7M.11A helper script: {path}")
    spec = importlib.util.spec_from_file_location("phase7m11a_helper_for_7m12a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    version = str(getattr(mod, "SCRIPT_VERSION", ""))
    if version != EXPECTED_7M11A_HELPER_VERSION:
        raise RuntimeError(
            f"Expected 7M.11A helper v{EXPECTED_7M11A_HELPER_VERSION}; found v{version or '?'} at {path}"
        )
    required = [
        "load_candidate_inputs", "load_primary_anchor_rows", "build_causal_clock",
        "MarkTimestampCache", "nearest_prior_surface_timestamps",
        "read_mark_rows_for_timestamps", "numericize_mark", "DayCache", "numericize_oi",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(f"7M.11A helper is missing required functions/classes: {missing}")
    return mod


# ---------------------------------------------------------------------------
# Candidate-level gamma landscape
# ---------------------------------------------------------------------------


def _latest_oi_per_symbol(oi: pd.DataFrame, target_ns: int, lookback_sec: int) -> pd.DataFrame:
    if oi.empty:
        return pd.DataFrame(columns=list(oi.columns))
    lo = int(target_ns - lookback_sec * NS_PER_SEC)
    w = oi[(oi["ts_recv_ns"] > lo) & (oi["ts_recv_ns"] <= target_ns)].copy()
    if w.empty:
        return w
    w = (
        w.sort_values(["symbol", "ts_recv_ns"], kind="mergesort")
        .groupby("symbol", as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    return w


class CausalOiCursor:
    """Incremental latest-per-contract OI state for monotonically increasing targets."""

    def __init__(self, oi: pd.DataFrame):
        self.frame = oi.sort_values(["ts_recv_ns", "symbol"], kind="mergesort").reset_index(drop=True).copy()
        self.times = self.frame["ts_recv_ns"].to_numpy(dtype=np.int64) if not self.frame.empty else np.array([], dtype=np.int64)
        self.symbols = self.frame["symbol"].astype(str).to_numpy() if not self.frame.empty else np.array([], dtype=object)
        self.pos = 0
        self.latest_idx: dict[str, int] = {}
        self.last_target = -1

    def latest(self, target_ns: int, lookback_sec: int) -> pd.DataFrame:
        target_ns = int(target_ns)
        if target_ns < self.last_target:
            raise RuntimeError("CausalOiCursor targets must be monotonically nondecreasing")
        self.last_target = target_ns
        while self.pos < len(self.times) and int(self.times[self.pos]) <= target_ns:
            self.latest_idx[str(self.symbols[self.pos])] = self.pos
            self.pos += 1
        lo = target_ns - int(lookback_sec * NS_PER_SEC)
        idxs = [i for i in self.latest_idx.values() if int(self.times[i]) > lo]
        if not idxs:
            return self.frame.iloc[0:0].copy()
        return self.frame.iloc[idxs].copy()


def gamma_landscape_metrics(
    mark_surface: pd.DataFrame,
    oi_window: pd.DataFrame,
    target_ns: int,
    mark_surface_ts_ns: int | None,
    oi_lookback_sec: int = OI_AUDIT_LOOKBACK_SEC,
) -> dict[str, Any]:
    """Compute unsigned observed OI x |gamma| geometry at one causal target."""
    out: dict[str, Any] = {
        "mark_surface_found": False,
        "mark_surface_ts_recv_ns": mark_surface_ts_ns,
        "mark_surface_age_sec": math.nan,
        "index_price_surface": math.nan,
        "mark_contracts": 0,
        "mark_expiries": 0,
        "mark_gamma_valid_frac": math.nan,
        "oi_contracts_recent": 0,
        "oi_expiries_recent": 0,
        "oi_age_median_sec": math.nan,
        "oi_age_p90_sec": math.nan,
        "oi_age_max_sec": math.nan,
        "oi_sum_recent": math.nan,
        "oi_value_sum_recent": math.nan,
        "oi_gamma_overlap_contracts": 0,
        "oi_gamma_overlap_contract_frac": math.nan,
        "oi_gamma_overlap_oi_share": math.nan,
        "oi_gamma_overlap_oi_value_share": math.nan,
        "gamma_oi_abs_total": math.nan,
        "gamma_oi_abs_delta_change_1pct_total": math.nan,
        "gamma_oi_abs_front_expiry": math.nan,
        "gamma_oi_abs_front_expiry_share": math.nan,
        "gamma_oi_abs_expiry_hhi": math.nan,
        "gamma_oi_abs_strike_hhi": math.nan,
        "gamma_oi_abs_dominant_strike_share": math.nan,
        "gamma_oi_abs_dominant_strike": math.nan,
        "gamma_oi_abs_dominant_strike_distance_bps": math.nan,
        "gamma_oi_abs_weighted_mean_abs_distance_bps": math.nan,
        "gamma_oi_abs_weighted_median_abs_distance_bps": math.nan,
        "gamma_oi_abs_weighted_mean_calendar_days": math.nan,
        "front_expiry_code": None,
        "front_calendar_days": math.nan,
        "gamma_oi_contracts": 0,
    }
    for band in NEAR_BANDS_PCT:
        tag = str(int(band))
        out[f"gamma_oi_abs_near_{tag}pct"] = math.nan
        out[f"gamma_oi_abs_near_{tag}pct_share"] = math.nan
        out[f"gamma_oi_abs_delta_change_1pct_near_{tag}pct"] = math.nan
        out[f"gamma_oi_contracts_near_{tag}pct"] = 0
        out[f"oi_recent_near_{tag}pct_sum"] = math.nan
        out[f"oi_gamma_overlap_near_{tag}pct_oi_share"] = math.nan
        out[f"oi_gamma_overlap_near_{tag}pct_contract_frac"] = math.nan

    if mark_surface_ts_ns is None or mark_surface.empty:
        return out

    out["mark_surface_found"] = True
    out["mark_surface_age_sec"] = float((int(target_ns) - int(mark_surface_ts_ns)) / NS_PER_SEC)

    mark = mark_surface.copy()
    forbid_outcome_columns(mark, "canonical mark/Greeks")
    idx = pd.to_numeric(mark.get("index_price"), errors="coerce")
    idx = idx[np.isfinite(idx) & (idx > 0)]
    if idx.empty:
        return out
    spot = float(idx.median())
    out["index_price_surface"] = spot

    mark = mark[mark.get("option_type").isin(["C", "P"])].copy()
    mark["strike"] = pd.to_numeric(mark.get("strike"), errors="coerce")
    mark["gamma"] = pd.to_numeric(mark.get("gamma"), errors="coerce")
    valid_contract = np.isfinite(mark["strike"]) & (mark["strike"] > 0)
    mark = mark[valid_contract].copy()
    if mark.empty:
        return out

    target_date = datetime.fromtimestamp(target_ns / NS_PER_SEC, tz=timezone.utc).date()
    mark["calendar_days"] = mark["expiry_date"].map(
        lambda x: (x - target_date).days if isinstance(x, date) else math.nan
    )
    mark = mark[pd.to_numeric(mark["calendar_days"], errors="coerce") >= 0].copy()
    if mark.empty:
        return out

    out["mark_contracts"] = int(mark["symbol"].nunique())
    out["mark_expiries"] = int(mark["expiry_code"].dropna().nunique())
    out["mark_gamma_valid_frac"] = float(np.isfinite(mark["gamma"]).mean()) if len(mark) else math.nan

    oi = _latest_oi_per_symbol(oi_window, target_ns, oi_lookback_sec)
    if oi.empty:
        return out
    forbid_outcome_columns(oi, "canonical options OI")
    oi["open_interest"] = pd.to_numeric(oi.get("open_interest"), errors="coerce")
    oi["open_interest_value"] = pd.to_numeric(oi.get("open_interest_value"), errors="coerce")
    oi = oi[np.isfinite(oi["open_interest"]) & (oi["open_interest"] >= 0)].copy()
    if oi.empty:
        return out

    ages = (int(target_ns) - oi["ts_recv_ns"].to_numpy(dtype=np.int64)) / NS_PER_SEC
    out["oi_contracts_recent"] = int(oi["symbol"].nunique())
    out["oi_expiries_recent"] = int(oi["expiry_code"].dropna().nunique())
    out["oi_age_median_sec"] = float(np.median(ages)) if len(ages) else math.nan
    out["oi_age_p90_sec"] = float(np.quantile(ages, 0.90)) if len(ages) else math.nan
    out["oi_age_max_sec"] = float(np.max(ages)) if len(ages) else math.nan
    out["oi_sum_recent"] = float(oi["open_interest"].sum())
    val = oi["open_interest_value"]
    out["oi_value_sum_recent"] = float(val[np.isfinite(val)].sum()) if np.isfinite(val).any() else math.nan

    # One row per mark symbol at the chosen exact surface.
    mark = mark.sort_values(["symbol", "ts_recv_ns"], kind="mergesort").groupby(
        "symbol", as_index=False, sort=False
    ).tail(1)

    joined = mark.merge(
        oi[["symbol", "ts_recv_ns", "open_interest", "open_interest_value"]].rename(
            columns={"ts_recv_ns": "oi_ts_recv_ns"}
        ),
        on="symbol", how="inner", validate="one_to_one",
    )
    if joined.empty:
        return out

    out["oi_gamma_overlap_contracts"] = int(joined["symbol"].nunique())
    out["oi_gamma_overlap_contract_frac"] = safe_div(
        float(out["oi_gamma_overlap_contracts"]), float(out["oi_contracts_recent"])
    )
    overlap_oi = float(pd.to_numeric(joined["open_interest"], errors="coerce").sum())
    out["oi_gamma_overlap_oi_share"] = safe_div(overlap_oi, float(out["oi_sum_recent"]))
    jval = pd.to_numeric(joined["open_interest_value"], errors="coerce")
    overlap_val = float(jval[np.isfinite(jval)].sum()) if np.isfinite(jval).any() else math.nan
    out["oi_gamma_overlap_oi_value_share"] = safe_div(overlap_val, float(out["oi_value_sum_recent"]))

    # Near-spot coverage is more relevant than exchange-wide overlap because mark/Greek
    # collection intentionally tracks a narrower, dynamically selected contract universe.
    oi["strike"] = pd.to_numeric(oi.get("strike"), errors="coerce")
    joined["strike"] = pd.to_numeric(joined.get("strike"), errors="coerce")
    for band in NEAR_BANDS_PCT:
        tag = str(int(band))
        oi_near = oi[np.isfinite(oi["strike"]) & (np.abs(oi["strike"] / spot - 1.0) * 100.0 <= band + 1e-12)]
        joined_near = joined[np.isfinite(joined["strike"]) & (np.abs(joined["strike"] / spot - 1.0) * 100.0 <= band + 1e-12)]
        near_oi_total = float(pd.to_numeric(oi_near["open_interest"], errors="coerce").sum()) if not oi_near.empty else 0.0
        near_overlap_oi = float(pd.to_numeric(joined_near["open_interest"], errors="coerce").sum()) if not joined_near.empty else 0.0
        out[f"oi_recent_near_{tag}pct_sum"] = near_oi_total
        out[f"oi_gamma_overlap_near_{tag}pct_oi_share"] = safe_div(near_overlap_oi, near_oi_total) if near_oi_total > 0 else math.nan
        out[f"oi_gamma_overlap_near_{tag}pct_contract_frac"] = safe_div(
            float(joined_near["symbol"].nunique()), float(oi_near["symbol"].nunique())
        ) if len(oi_near) else math.nan

    joined["gamma"] = pd.to_numeric(joined["gamma"], errors="coerce")
    joined["open_interest"] = pd.to_numeric(joined["open_interest"], errors="coerce")
    joined = joined[
        np.isfinite(joined["gamma"]) & np.isfinite(joined["open_interest"]) &
        (joined["open_interest"] >= 0)
    ].copy()
    if joined.empty:
        return out

    joined["gamma_oi_abs"] = joined["open_interest"] * np.abs(joined["gamma"])
    joined = joined[np.isfinite(joined["gamma_oi_abs"]) & (joined["gamma_oi_abs"] >= 0)].copy()
    if joined.empty:
        return out

    joined["abs_distance_pct"] = np.abs(joined["strike"] / spot - 1.0) * 100.0
    joined["abs_distance_bps"] = joined["abs_distance_pct"] * 100.0
    joined["delta_change_1pct_proxy"] = joined["gamma_oi_abs"] * spot * 0.01

    total = float(joined["gamma_oi_abs"].sum())
    total_1pct = float(joined["delta_change_1pct_proxy"].sum())
    out["gamma_oi_contracts"] = int(joined["symbol"].nunique())
    out["gamma_oi_abs_total"] = total
    out["gamma_oi_abs_delta_change_1pct_total"] = total_1pct

    for band in NEAR_BANDS_PCT:
        tag = str(int(band))
        w = joined[joined["abs_distance_pct"] <= band + 1e-12]
        s = float(w["gamma_oi_abs"].sum())
        s1 = float(w["delta_change_1pct_proxy"].sum())
        out[f"gamma_oi_abs_near_{tag}pct"] = s
        out[f"gamma_oi_abs_near_{tag}pct_share"] = safe_div(s, total)
        out[f"gamma_oi_abs_delta_change_1pct_near_{tag}pct"] = s1
        out[f"gamma_oi_contracts_near_{tag}pct"] = int(w["symbol"].nunique())

    if total > 0:
        expiry_grp = joined.groupby("expiry_code", dropna=True)["gamma_oi_abs"].sum()
        strike_grp = joined.groupby("strike", dropna=True)["gamma_oi_abs"].sum()
        out["gamma_oi_abs_expiry_hhi"] = hhi(expiry_grp.values)
        out["gamma_oi_abs_strike_hhi"] = hhi(strike_grp.values)

        if len(strike_grp):
            dom_strike = float(strike_grp.idxmax())
            dom_val = float(strike_grp.max())
            out["gamma_oi_abs_dominant_strike"] = dom_strike
            out["gamma_oi_abs_dominant_strike_share"] = dom_val / total
            out["gamma_oi_abs_dominant_strike_distance_bps"] = abs(dom_strike / spot - 1.0) * 10_000.0

        weights = joined["gamma_oi_abs"].to_numpy(dtype=float)
        dist_bps = joined["abs_distance_bps"].to_numpy(dtype=float)
        out["gamma_oi_abs_weighted_mean_abs_distance_bps"] = float(np.average(dist_bps, weights=weights))
        out["gamma_oi_abs_weighted_median_abs_distance_bps"] = weighted_quantile(dist_bps, weights, 0.5)
        days = pd.to_numeric(joined["calendar_days"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(days) & np.isfinite(weights) & (weights > 0)
        if mask.any():
            out["gamma_oi_abs_weighted_mean_calendar_days"] = float(np.average(days[mask], weights=weights[mask]))

        expiry_order = (
            joined[["expiry_code", "calendar_days"]].dropna().drop_duplicates()
            .sort_values(["calendar_days", "expiry_code"], kind="mergesort")
        )
        if not expiry_order.empty:
            front_code = str(expiry_order.iloc[0]["expiry_code"])
            front_days = float(expiry_order.iloc[0]["calendar_days"])
            front = joined[joined["expiry_code"].astype(str) == front_code]
            front_total = float(front["gamma_oi_abs"].sum())
            out["front_expiry_code"] = front_code
            out["front_calendar_days"] = front_days
            out["gamma_oi_abs_front_expiry"] = front_total
            out["gamma_oi_abs_front_expiry_share"] = front_total / total

    return out


# ---------------------------------------------------------------------------
# Per-day checkpointing
# ---------------------------------------------------------------------------


def checkpoint_paths(output_root: Path, day_text: str) -> tuple[Path, Path]:
    root = output_root / "day_checkpoints" / f"candidate_date={day_text}"
    return root / "part.parquet", root / "manifest.json"


def load_checkpoint(output_root: Path, day_text: str) -> pd.DataFrame | None:
    part, manifest = checkpoint_paths(output_root, day_text)
    if not (part.exists() and manifest.exists()):
        return None
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(meta.get("contract_version")) != CHECKPOINT_CONTRACT_VERSION:
        return None
    if str(meta.get("candidate_date")) != day_text:
        return None
    frame = pd.read_parquet(part)
    forbid_outcome_columns(frame, "7M.12A checkpoint")
    if len(frame) != int(meta.get("candidate_rows", -1)):
        return None
    return frame


def save_checkpoint(output_root: Path, day_text: str, frame: pd.DataFrame) -> None:
    part, manifest = checkpoint_paths(output_root, day_text)
    atomic_write_parquet(frame, part)
    write_json(manifest, {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "contract_version": CHECKPOINT_CONTRACT_VERSION,
        "candidate_date": day_text,
        "candidate_rows": int(len(frame)),
        "created_at_utc": utc_now_iso(),
        "part_sha256": sha256_file(part),
    })


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


PRIMARY_METRICS = [
    "gamma_oi_abs_total",
    "gamma_oi_abs_delta_change_1pct_total",
    "gamma_oi_abs_near_1pct",
    "gamma_oi_abs_near_2pct",
    "gamma_oi_abs_near_5pct",
    "gamma_oi_abs_delta_change_1pct_near_1pct",
    "gamma_oi_abs_delta_change_1pct_near_2pct",
    "gamma_oi_abs_delta_change_1pct_near_5pct",
    "gamma_oi_abs_near_1pct_share",
    "gamma_oi_abs_near_2pct_share",
    "gamma_oi_abs_near_5pct_share",
    "gamma_oi_abs_front_expiry_share",
    "gamma_oi_abs_expiry_hhi",
    "gamma_oi_abs_strike_hhi",
    "gamma_oi_abs_dominant_strike_share",
    "gamma_oi_abs_dominant_strike_distance_bps",
    "gamma_oi_abs_weighted_mean_abs_distance_bps",
    "gamma_oi_abs_weighted_median_abs_distance_bps",
    "gamma_oi_abs_weighted_mean_calendar_days",
]


def metric_distribution_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    groups: list[tuple[str, pd.DataFrame]] = [("FULL", frame)]
    for period in ("DISCOVERY", "VALIDATION"):
        groups.append((period, frame[frame["period"] == period]))
    groups.append(("UP", frame[frame["direction"] == 1]))
    groups.append(("DOWN", frame[frame["direction"] == -1]))

    rows: list[dict[str, Any]] = []
    for group_name, g in groups:
        for metric in PRIMARY_METRICS:
            if metric not in g.columns:
                continue
            x = pd.to_numeric(g[metric], errors="coerce")
            finite = x[np.isfinite(x)]
            stats = qstats(finite)
            row = {
                "group": group_name,
                "metric": metric,
                "rows": int(len(g)),
                "valid": int(len(finite)),
                "valid_frac": float(len(finite) / len(g)) if len(g) else math.nan,
                "mean": float(finite.mean()) if len(finite) else math.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else math.nan,
                "nunique": int(finite.nunique()) if len(finite) else 0,
            }
            row.update(stats)
            rows.append(row)
    return rows


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in PRIMARY_METRICS if m in frame.columns]
    data = frame[metrics].apply(pd.to_numeric, errors="coerce")
    # log1p only for nonnegative scale metrics; shares/concentration/distance stay raw.
    for c in list(data.columns):
        if c.startswith("gamma_oi_abs_total") or "near_" in c and not c.endswith("_share"):
            vals = data[c]
            if (vals.dropna() >= 0).all():
                data[c] = np.log1p(vals)
    corr = data.corr(method="spearman", min_periods=50)
    out = corr.reset_index().rename(columns={"index": "metric"})
    return out


def date_direction_variation(frame: pd.DataFrame) -> pd.DataFrame:
    metric = "gamma_oi_abs_delta_change_1pct_near_2pct"
    rows = []
    for (d, direction), g in frame.groupby(["candidate_date", "direction"], sort=True):
        x = pd.to_numeric(g[metric], errors="coerce")
        x = x[np.isfinite(x)]
        q = qstats(x)
        rows.append({
            "candidate_date": d,
            "direction": int(direction),
            "rows": int(len(g)),
            "valid": int(len(x)),
            "nunique": int(x.nunique()) if len(x) else 0,
            "p25": q["p25"],
            "median": q["median"],
            "p75": q["p75"],
            "iqr": q["p75"] - q["p25"] if math.isfinite(q["p75"]) and math.isfinite(q["p25"]) else math.nan,
        })
    return pd.DataFrame(rows)


def build_reports(frame: pd.DataFrame, report_root: Path, summary: dict[str, Any]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)

    dist = pd.DataFrame(metric_distribution_rows(frame))
    write_csv(report_root / "gamma_metric_distribution.csv", dist)
    write_csv(report_root / "gamma_metric_correlations.csv", correlation_table(frame))
    write_csv(report_root / "date_direction_variation.csv", date_direction_variation(frame))

    day_rows = []
    for d, g in frame.groupby("candidate_date", sort=True):
        valid = pd.to_numeric(g["gamma_oi_abs_total"], errors="coerce")
        valid_mask = np.isfinite(valid)
        day_rows.append({
            "candidate_date": d,
            "candidates": int(len(g)),
            "anchor_valid_3s": int(g["anchor_valid_3s"].fillna(False).astype(bool).sum()),
            "mark_surface_found": int(g["mark_surface_found"].fillna(False).astype(bool).sum()),
            "gamma_landscape_valid": int(valid_mask.sum()),
            "gamma_landscape_valid_frac": float(valid_mask.mean()) if len(g) else math.nan,
            "median_mark_surface_age_sec": float(pd.to_numeric(g["mark_surface_age_sec"], errors="coerce").median()),
            "median_oi_age_sec": float(pd.to_numeric(g["oi_age_median_sec"], errors="coerce").median()),
            "median_overlap_oi_share": float(pd.to_numeric(g["oi_gamma_overlap_oi_share"], errors="coerce").median()),
            "median_near2_1pct_delta_change_proxy": float(
                pd.to_numeric(g["gamma_oi_abs_delta_change_1pct_near_2pct"], errors="coerce").median()
            ),
        })
    write_csv(report_root / "day_coverage.csv", pd.DataFrame(day_rows))

    subgroup_rows = []
    for period in ("DISCOVERY", "VALIDATION"):
        for direction in (-1, 1):
            g = frame[(frame["period"] == period) & (frame["direction"] == direction)]
            valid = np.isfinite(pd.to_numeric(g["gamma_oi_abs_total"], errors="coerce"))
            subgroup_rows.append({
                "period": period,
                "direction": direction,
                "candidates": int(len(g)),
                "gamma_landscape_valid": int(valid.sum()),
                "valid_frac": float(valid.mean()) if len(g) else math.nan,
            })
    write_csv(report_root / "subgroup_representation.csv", pd.DataFrame(subgroup_rows))

    overlap_cols = [
        "oi_contracts_recent", "mark_contracts", "oi_gamma_overlap_contracts",
        "oi_gamma_overlap_contract_frac", "oi_gamma_overlap_oi_share",
        "oi_gamma_overlap_oi_value_share", "oi_age_median_sec", "oi_age_p90_sec",
        "mark_surface_age_sec", "mark_gamma_valid_frac",
        "oi_gamma_overlap_near_1pct_oi_share", "oi_gamma_overlap_near_2pct_oi_share",
        "oi_gamma_overlap_near_5pct_oi_share",
    ]
    overlap_rows = []
    for col in overlap_cols:
        x = pd.to_numeric(frame[col], errors="coerce")
        finite = x[np.isfinite(x)]
        row = {"metric": col, "valid": int(len(finite)), "valid_frac": float(len(finite)/len(frame)) if len(frame) else math.nan}
        row.update(qstats(finite))
        overlap_rows.append(row)
    write_csv(report_root / "gamma_oi_overlap_summary.csv", pd.DataFrame(overlap_rows))

    invariants = pd.DataFrame([
        {"invariant": "phase", "value": PHASE},
        {"invariant": "script_version", "value": SCRIPT_VERSION},
        {"invariant": "outcomes_opened", "value": "NO"},
        {"invariant": "causal_cutoff", "value": "start of candidate_second"},
        {"invariant": "mark_surface_selection", "value": "exact latest canonical ts_recv_ns <= causal cutoff"},
        {"invariant": "oi_selection", "value": "latest per contract within preceding 1800s audit ceiling"},
        {"invariant": "oi_lookback_status", "value": "FEASIBILITY_AUDIT_ONLY_NOT_FROZEN_SCIENCE_THRESHOLD"},
        {"invariant": "gamma_sign", "value": "absolute gamma only; no dealer/net sign inferred"},
        {"invariant": "primary_quantity", "value": "open_interest * abs(gamma)"},
        {"invariant": "scale_normalized_quantity", "value": "open_interest * abs(gamma) * spot * 0.01"},
        {"invariant": "near_spot_bands", "value": "+/-1%, +/-2%, +/-5% descriptive only"},
        {"invariant": "universe_scope", "value": "contracts with collected mark/Greeks gamma joined to recent expiry-scoped OI"},
        {"invariant": "dealer_gamma_claim", "value": "PROHIBITED"},
        {"invariant": "option_trade_direction_used", "value": "NO"},
        {"invariant": "iv_outcome_used_to_define_exposure", "value": "NO"},
    ])
    write_csv(report_root / "analysis_invariants.csv", invariants)

    valid = np.isfinite(pd.to_numeric(frame["gamma_oi_abs_total"], errors="coerce"))
    overlap = pd.to_numeric(frame["oi_gamma_overlap_oi_share"], errors="coerce")
    mark_age = pd.to_numeric(frame["mark_surface_age_sec"], errors="coerce")
    oi_age = pd.to_numeric(frame["oi_age_median_sec"], errors="coerce")
    near2 = pd.to_numeric(frame["gamma_oi_abs_delta_change_1pct_near_2pct"], errors="coerce")

    findings = "# Phase 7M.12A — outcome-blind options gamma-landscape feasibility\n\n"
    findings += "## Question\n\n"
    findings += "Immediately before the perp candidate starts, how much **observed unsigned OI-weighted gamma sensitivity** is near spot, and is the substrate usable for later matching?\n\n"
    findings += "## Guardrails\n\n"
    findings += "- No future outcome file was opened.\n"
    findings += "- Exposure is `OI × |gamma|`; no participant/dealer long-short sign is inferred.\n"
    findings += "- The 30-minute OI lookback is only a broad feasibility ceiling, not a frozen later threshold.\n"
    findings += "- ±1/2/5% strike bands are descriptive geometry, not tuned exposure cutoffs.\n"
    findings += "- Aggregates cover the mark/Greeks contracts for which contemporaneous gamma was actually collected; OI overlap is reported explicitly.\n\n"
    findings += "## Coverage snapshot\n\n"
    findings += f"- Frozen candidates: **{len(frame):,}**\n"
    findings += f"- Valid gamma landscapes: **{int(valid.sum()):,} ({float(valid.mean())*100:.2f}%)**\n"
    findings += f"- Median mark-surface age: **{float(mark_age.median()):.3f}s**\n"
    findings += f"- Median per-candidate OI age median: **{float(oi_age.median()):.3f}s**\n"
    findings += f"- Median share of recent OI represented by mark/Greek-overlap contracts: **{float(overlap.median())*100:.2f}%**\n"
    findings += f"- Near-2% 1%-move delta-change proxy median: **{float(near2.median()):.8g}**\n\n"
    findings += "## Interpretation\n\n"
    findings += "A high gamma-sensitivity value means a larger amount of **observed option delta sensitivity could change when BTC moves**. It does not say whether hedgers would buy or sell BTC.\n\n"
    findings += "Use `gamma_metric_distribution.csv`, `date_direction_variation.csv`, and `gamma_metric_correlations.csv` to decide which single unsigned gamma primitive has enough coverage, within-stratum variation, and non-redundancy to freeze for 7M.12B matching.\n"
    (report_root / "findings.md").write_text(findings, encoding="utf-8")

    write_json(report_root / "summary.json", summary)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_stack()
    helper = load_7m11a_helper(args.helper_script)

    candidates = helper.load_candidate_inputs(args.candidate_root)
    anchors = helper.load_primary_anchor_rows(args.sequence_root)
    clock = helper.build_causal_clock(candidates, anchors)
    forbid_outcome_columns(clock, "frozen candidate clock")
    if len(clock) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_CANDIDATES:,} candidates; found {len(clock):,}")

    # Preserve the full frozen universe; invalid anchors remain explicit missing rows.
    clock = clock.copy()
    clock["candidate_date"] = clock["candidate_date"].astype(str)
    clock["period"] = clock["candidate_date"].map(split_period)

    mark_ts_cache = helper.MarkTimestampCache(args.canonical_root, max_days=3)
    oi_cache = helper.DayCache(args.canonical_root, max_days=2)

    all_days = sorted(clock["candidate_date"].unique())
    parts: list[pd.DataFrame] = []
    day_summaries: list[dict[str, Any]] = []

    print(f"{PHASE} - Outcome-blind BTC-options gamma-landscape feasibility")
    print("[1/4] Verifying frozen candidate clock and causal options inputs...")
    print("[2/4] Auditing pre-event OI-weighted absolute gamma landscape...")

    for day_idx, day_text in enumerate(all_days, start=1):
        day_frame = clock[clock["candidate_date"] == day_text].copy().sort_values(
            ["pre_event_cutoff_ns", "candidate_id"], kind="mergesort"
        )
        cached = None if args.force else load_checkpoint(args.output_root, day_text)
        if cached is not None:
            parts.append(cached)
            valid_cached = np.isfinite(pd.to_numeric(cached.get("gamma_oi_abs_total"), errors="coerce"))
            overlap_cached = pd.to_numeric(cached.get("oi_gamma_overlap_oi_share"), errors="coerce")
            day_summaries.append({
                "candidate_date": day_text,
                "candidates": int(len(cached)),
                "checkpoint_reused": True,
                "valid_gamma_landscapes": int(valid_cached.sum()),
                "median_overlap_oi_share": safe_float(overlap_cached.median()),
            })
            print(
                f"  [{day_idx:02d}/{len(all_days):02d}] {day_text}: candidates={len(day_frame):4d} "
                f"valid={int(valid_cached.sum()):4d} checkpoint=REUSED"
            )
            continue

        day = date.fromisoformat(day_text)
        valid_targets = day_frame.loc[
            day_frame["anchor_valid_3s"].fillna(False).astype(bool), "pre_event_cutoff_ns"
        ].astype("int64").tolist()

        unique_ts_window, _cur_ts, mark_rows = mark_ts_cache.load_window(day)
        resolved = helper.nearest_prior_surface_timestamps(unique_ts_window, valid_targets)
        needed_surface_ts = sorted({int(v) for v in resolved.values() if v is not None})
        mark_rows_targeted = helper.read_mark_rows_for_timestamps(
            args.canonical_root,
            [day - timedelta(days=1), day],
            needed_surface_ts,
            MARK_COLUMNS,
        )
        mark_rows_targeted = helper.numericize_mark(mark_rows_targeted)
        if not mark_rows_targeted.empty:
            mark_groups = {int(ts): g.copy() for ts, g in mark_rows_targeted.groupby("ts_recv_ns", sort=False)}
        else:
            mark_groups = {}

        oi = oi_cache.load_window("open_interest", day, OI_COLUMNS, include_prev=True)
        oi = helper.numericize_oi(oi)
        oi_cursor = CausalOiCursor(oi)

        rows: list[dict[str, Any]] = []
        for r in day_frame.itertuples(index=False):
            base = {
                "candidate_id": str(r.candidate_id),
                "candidate_date": str(r.candidate_date),
                "period": str(r.period),
                "direction": int(r.direction),
                "candidate_second": int(r.candidate_second),
                "pre_event_cutoff_ns": int(r.pre_event_cutoff_ns),
                "anchor_valid_3s": bool(r.anchor_valid_3s),
                "anchor_ts_recv_ns": safe_float(getattr(r, "anchor_ts_recv_ns", math.nan)),
            }
            if not base["anchor_valid_3s"]:
                metrics = gamma_landscape_metrics(
                    pd.DataFrame(), oi.iloc[0:0].copy(), base["pre_event_cutoff_ns"], None, OI_AUDIT_LOOKBACK_SEC
                )
                base.update(metrics)
                rows.append(base)
                continue

            target = int(base["pre_event_cutoff_ns"])
            surface_ts = resolved.get(target)
            surface = mark_groups.get(int(surface_ts), pd.DataFrame()) if surface_ts is not None else pd.DataFrame()
            oi_latest = oi_cursor.latest(target, OI_AUDIT_LOOKBACK_SEC)
            metrics = gamma_landscape_metrics(surface, oi_latest, target, surface_ts, OI_AUDIT_LOOKBACK_SEC)
            base.update(metrics)
            rows.append(base)

        out_day = pd.DataFrame(rows)
        forbid_outcome_columns(out_day, "7M.12A candidate audit")
        save_checkpoint(args.output_root, day_text, out_day)
        parts.append(out_day)

        valid = np.isfinite(pd.to_numeric(out_day["gamma_oi_abs_total"], errors="coerce"))
        overlap_med = pd.to_numeric(out_day["oi_gamma_overlap_oi_share"], errors="coerce").median()
        day_summaries.append({
            "candidate_date": day_text,
            "candidates": int(len(out_day)),
            "canonical_mark_rows": int(mark_rows),
            "needed_mark_surfaces": int(len(needed_surface_ts)),
            "loaded_mark_rows": int(len(mark_rows_targeted)),
            "valid_gamma_landscapes": int(valid.sum()),
            "median_overlap_oi_share": safe_float(overlap_med),
        })
        print(
            f"  [{day_idx:02d}/{len(all_days):02d}] {day_text}: candidates={len(out_day):4d} "
            f"mark_rows={mark_rows:8d} surfaces={len(needed_surface_ts):4d} "
            f"loaded_rows={len(mark_rows_targeted):6d} valid={int(valid.sum()):4d} "
            f"oi_overlap_med={safe_float(overlap_med):.3f} checkpoint=SAVED"
        )

    audit = pd.concat(parts, ignore_index=True)
    if len(audit) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_CANDIDATES:,} audit rows; found {len(audit):,}")
    if audit["candidate_id"].duplicated().any():
        raise RuntimeError("7M.12A audit contains duplicate candidate_id")
    forbid_outcome_columns(audit, "7M.12A final audit")

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / "candidate_options_gamma_landscape_audit.parquet"
    atomic_write_parquet(audit, out_path)

    print("[3/4] Writing outcome-blind feasibility reports...")
    valid = np.isfinite(pd.to_numeric(audit["gamma_oi_abs_total"], errors="coerce"))
    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_status": "OUTCOME_BLIND_FEASIBILITY_ONLY",
        "candidate_count": int(len(audit)),
        "anchor_valid_3s_count": int(audit["anchor_valid_3s"].fillna(False).astype(bool).sum()),
        "gamma_landscape_valid_count": int(valid.sum()),
        "gamma_landscape_valid_frac": float(valid.mean()),
        "primary_quantity": "open_interest * abs(gamma)",
        "scale_normalized_quantity": "open_interest * abs(gamma) * spot_index_price * 0.01",
        "near_spot_bands_pct": list(NEAR_BANDS_PCT),
        "oi_audit_lookback_sec": OI_AUDIT_LOOKBACK_SEC,
        "oi_lookback_is_frozen_scientific_threshold": False,
        "dealer_or_net_gamma_sign_inferred": False,
        "future_outcomes_opened": False,
        "candidate_audit_path": str(out_path),
        "candidate_audit_sha256": sha256_file(out_path),
        "helper_script": str(args.helper_script),
        "helper_script_sha256": sha256_file(args.helper_script),
        "day_summaries": day_summaries,
    }
    build_reports(audit, args.report_root, summary)

    print("[4/4] Complete.")
    print(f"  valid gamma landscapes: {int(valid.sum()):,}/{len(audit):,} ({float(valid.mean())*100:.2f}%)")
    print(f"  candidate audit: {out_path}")
    print(f"  reports:         {args.report_root}")
    return audit, summary


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """Pure-Python/pandas scientific-contract test; does not require project data."""
    target = 2_000_000_000_000
    spot = 100.0
    surface_ts = target - 500_000_000
    mark = pd.DataFrame([
        {"ts_recv_ns": surface_ts, "symbol": "BTC-260101-100-C", "index_price": spot, "strike": 100.0, "gamma": 0.020, "expiry_code": "260101", "expiry_date": date(2026,1,1), "option_type": "C"},
        {"ts_recv_ns": surface_ts, "symbol": "BTC-260101-100-P", "index_price": spot, "strike": 100.0, "gamma": 0.020, "expiry_code": "260101", "expiry_date": date(2026,1,1), "option_type": "P"},
        {"ts_recv_ns": surface_ts, "symbol": "BTC-260201-105-C", "index_price": spot, "strike": 105.0, "gamma": 0.010, "expiry_code": "260201", "expiry_date": date(2026,2,1), "option_type": "C"},
    ])
    # target date above is 1970 if interpreted literally; use a realistic 2025 timestamp.
    target_dt = datetime(2025,12,15,tzinfo=timezone.utc)
    target = int(target_dt.timestamp() * NS_PER_SEC)
    surface_ts = target - 500_000_000
    mark["ts_recv_ns"] = surface_ts
    oi = pd.DataFrame([
        {"ts_recv_ns": target - 10*NS_PER_SEC, "symbol": "BTC-260101-100-C", "open_interest": 100.0, "open_interest_value": 10.0, "expiry_code": "260101", "expiry_date": date(2026,1,1), "strike":100.0, "option_type":"C"},
        {"ts_recv_ns": target - 20*NS_PER_SEC, "symbol": "BTC-260101-100-P", "open_interest": 50.0, "open_interest_value": 5.0, "expiry_code": "260101", "expiry_date": date(2026,1,1), "strike":100.0, "option_type":"P"},
        {"ts_recv_ns": target - 30*NS_PER_SEC, "symbol": "BTC-260201-105-C", "open_interest": 40.0, "open_interest_value": 4.0, "expiry_code": "260201", "expiry_date": date(2026,2,1), "strike":105.0, "option_type":"C"},
        # OI-only contract deliberately outside mark/Greek overlap.
        {"ts_recv_ns": target - 40*NS_PER_SEC, "symbol": "BTC-260201-130-C", "open_interest": 10.0, "open_interest_value": 1.0, "expiry_code": "260201", "expiry_date": date(2026,2,1), "strike":130.0, "option_type":"C"},
    ])
    result = gamma_landscape_metrics(mark, oi, target, surface_ts, 1800)
    expected_total = 100*0.020 + 50*0.020 + 40*0.010
    assert abs(result["gamma_oi_abs_total"] - expected_total) < 1e-12
    assert abs(result["gamma_oi_abs_near_1pct"] - 3.0) < 1e-12
    assert abs(result["gamma_oi_abs_near_5pct"] - expected_total) < 1e-12
    assert result["oi_gamma_overlap_contracts"] == 3
    assert abs(result["oi_gamma_overlap_oi_share"] - 190/200) < 1e-12
    assert abs(result["oi_gamma_overlap_near_5pct_oi_share"] - 1.0) < 1e-12
    assert abs(result["gamma_oi_abs_delta_change_1pct_total"] - expected_total*spot*0.01) < 1e-12
    assert result["front_expiry_code"] == "260101"
    assert abs(result["gamma_oi_abs_front_expiry"] - 3.0) < 1e-12
    assert result["gamma_oi_abs_dominant_strike"] == 100.0
    assert result["gamma_oi_abs_dominant_strike_distance_bps"] == 0.0

    # Future OI must not enter latest causal state.
    oi_future = pd.concat([oi, pd.DataFrame([{
        "ts_recv_ns": target + NS_PER_SEC,
        "symbol": "BTC-260101-100-C", "open_interest": 999999.0, "open_interest_value": 999999.0,
        "expiry_code": "260101", "expiry_date": date(2026,1,1), "strike":100.0, "option_type":"C",
    }])], ignore_index=True)
    result_future = gamma_landscape_metrics(mark, oi_future, target, surface_ts, 1800)
    assert abs(result_future["gamma_oi_abs_total"] - expected_total) < 1e-12

    cursor = CausalOiCursor(oi_future)
    snap = cursor.latest(target, 1800)
    assert float(snap.loc[snap["symbol"] == "BTC-260101-100-C", "open_interest"].iloc[0]) == 100.0
    later = cursor.latest(target + 2*NS_PER_SEC, 1800)
    assert float(later.loc[later["symbol"] == "BTC-260101-100-C", "open_interest"].iloc[0]) == 999999.0

    print(
        "SELF-TEST PASS: unsigned OI×|gamma| aggregation, 1% move normalization, "
        "near-spot geometry, OI/Greek overlap accounting, expiry/strike concentration, "
        "and no-future OI selection validated."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--helper-script", type=Path, default=DEFAULT_7M11A_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--force", action="store_true", help="Ignore reusable per-day checkpoints and recompute.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
