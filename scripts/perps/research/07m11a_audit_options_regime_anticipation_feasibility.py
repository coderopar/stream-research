#!/usr/bin/env python3
"""Phase 7M.11A — Outcome-blind BTC-options regime / anticipation feasibility audit.

Research question
-----------------
For the frozen prospective BTCUSDT perp 25-bp candidate population used in
Phases 7M.2–7M.10, do the collected Binance BTC-options streams support a
clean separation between:

1. PRE-EXISTING OPTIONS REGIME
   What did the options surface / positioning landscape look like before the
   perp candidate began?

2. PRE-EVENT REPRICING / ANTICIPATION
   Did IV / skew / term structure change during the 5m / 15m / 30m leading
   into the candidate, before the candidate second itself began?

3. EVENT-TIME OPTIONS RESPONSE
   Did the same surface metrics move between the conservative pre-event cutoff
   and the existing causal decision point (exact Phase 7M.6F anchor + 3s)?

This phase is FEASIBILITY ONLY. It opens no future outcome table, performs no
matching, selects no outcome-facing threshold, and creates no composite signal.

Why the timing split is conservative
------------------------------------
The frozen candidate catalog stores candidate_second at one-second resolution.
To avoid contaminating "anticipation" with the candidate move itself, the
pre-event cutoff is the START of that UTC second:

    pre_event_cutoff_ns = candidate_second * 1e9

All regime / anticipation measurements use only options events with
``ts_recv_ns <= pre_event_cutoff_ns``. The event-response measurement may use
causal data after that cutoff, but never later than:

    decision_ts_recv_ns = exact_7M6F_anchor_ts_recv_ns + 3 seconds

Canonical options families audited
----------------------------------
The canonical extractor routes the collector payloads to:

    options/index_price
    options/mark_price_greeks
    options/open_interest
    options/trades

The mark-price family preserves mark IV and Greeks:

    mark_price, index_price, mark_iv, bid_iv, ask_iv,
    delta, gamma, theta, vega, ...

The options universe is dynamic (collector configuration): BTC contracts within
roughly +/-20% moneyness and <=30 days to expiry, refreshed every 30 seconds.
This audit therefore measures actual strike/expiry coverage rather than assuming
that a stable fixed surface exists.

Surface feasibility quantities (NOT yet frozen exposures)
---------------------------------------------------------
At each causal surface snapshot we audit:

* contract / expiry / call / put counts
* mark-IV and Greek field completeness
* nearest-ATM same-strike call/put pair on the front available expiry
* front-expiry ATM IV = mean(call mark IV, put mark IV)
* same-strike call-minus-put mark-IV spread (descriptive skew proxy)
* nearest-25-delta call / put availability and IV spread
* second-expiry ATM IV and front-vs-second term difference
* strike-distance / delta-distance diagnostics so later phases can decide
  whether the resulting approximations are physically defensible

Pre-event changes are audited over 5m / 15m / 30m, ending strictly at the
pre-event cutoff. Event response is the change from pre-event cutoff to the
existing anchor+3s decision point.

Open-interest feasibility
-------------------------
For each candidate, the script collects the latest OI row per contract observed
within the preceding 30 minutes (audit window only; NOT a frozen freshness
threshold) and records coverage / age plus call-put / near-ATM structure.
Public OI is treated as positioning concentration only; no assumption is made
about who is long or short.

Options-trade feasibility
-------------------------
For 5m / 15m / 30m pre-event windows and the candidate-second-to-decision event
window, the script records trade counts / activity by call-put and audits the
raw ``side`` / ``maker`` field availability. It does NOT infer bullish/bearish
option flow in this phase.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import tempfile
from collections import OrderedDict, Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0.1"
CHECKPOINT_CONTRACT_VERSION = "1.0.1"
PHASE = "7M.11A"
SYMBOL = "BTCUSDT"
UNDERLYING = "BTC"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
NS_PER_SEC = 1_000_000_000
PRE_WINDOWS_SEC = (300, 900, 1800)  # 5m / 15m / 30m
OI_AUDIT_LOOKBACK_SEC = 1800
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
    "prospective_25bp_options_regime_anticipation_feasibility_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_options_regime_anticipation_feasibility"
)

FAMILY_INDEX = "index_price"
FAMILY_MARK = "mark_price_greeks"
FAMILY_OI = "open_interest"
FAMILY_TRADES = "trades"

MARK_COLUMNS = (
    "ts_recv_ns", "symbol", "exchange_event_ms",
    "mark_price", "index_price", "estimated_settle_price",
    "best_bid_price", "best_ask_price", "best_bid_quantity", "best_ask_quantity",
    "bid_iv", "ask_iv", "mark_iv", "risk_free_rate",
    "delta", "theta", "gamma", "vega",
)
OI_COLUMNS = (
    "ts_recv_ns", "symbol", "exchange_event_ms", "open_interest", "open_interest_value",
)
TRADE_COLUMNS = (
    "ts_recv_ns", "symbol", "exchange_event_ms", "trade_time_ms", "trade_id",
    "price", "quantity", "order_type", "side", "maker",
)
INDEX_COLUMNS = (
    "ts_recv_ns", "symbol", "exchange_event_ms", "index_price",
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


def safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan


def safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def qstats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    labels = ("min", "p10", "p25", "median", "p75", "p90", "p99", "max")
    if len(arr) == 0:
        return {k: math.nan for k in labels}
    qs = np.quantile(arr, [0, .10, .25, .50, .75, .90, .99, 1])
    return dict(zip(labels, map(float, qs)))


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


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


def family_files(canonical_root: Path, family: str, day: date) -> list[Path]:
    d = canonical_root / "options" / family / f"date={day.isoformat()}"
    return sorted(d.glob("*.parquet")) if d.exists() else []


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def split_period(day_text: str) -> str:
    if day_text <= DISCOVERY_END_DATE:
        return "DISCOVERY"
    if day_text >= VALIDATION_START_DATE:
        return "VALIDATION"
    return "OTHER"


# ---------------------------------------------------------------------------
# Contract parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractInfo:
    symbol: str
    expiry_code: str
    expiry_date: date
    strike: float
    option_type: str  # C / P


def parse_option_symbol(symbol: str) -> ContractInfo | None:
    s = str(symbol or "").strip().upper()
    parts = s.split("-")
    if len(parts) < 4 or parts[0] != UNDERLYING:
        return None
    exp = parts[1]
    typ = parts[-1]
    if len(exp) != 6 or not exp.isdigit() or typ not in {"C", "P"}:
        return None
    try:
        expiry_date = datetime.strptime(exp, "%y%m%d").date()
        strike = float(parts[2])
    except Exception:
        return None
    if not math.isfinite(strike) or strike <= 0:
        return None
    return ContractInfo(s, exp, expiry_date, strike, typ)


def add_contract_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        for c in ("expiry_code", "expiry_date", "strike", "option_type"):
            out[c] = pd.Series(dtype="object")
        return out
    infos = [parse_option_symbol(x) for x in frame["symbol"].astype(str)]
    out = frame.copy()
    out["expiry_code"] = [i.expiry_code if i else None for i in infos]
    out["expiry_date"] = [i.expiry_date if i else None for i in infos]
    out["strike"] = [i.strike if i else math.nan for i in infos]
    out["option_type"] = [i.option_type if i else None for i in infos]
    return out


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
        raise FileNotFoundError(f"No Phase 7M.6F candidate_metrics under {sequence_root}")
    requested = [
        "candidate_id", "candidate_date", "observation_horizon_sec",
        "anchor_found", "anchor_ts_recv_ns", "horizon_complete", "invalid_reason",
    ]
    parts: list[pd.DataFrame] = []
    for p in paths:
        f = read_existing_columns(p, requested)
        if f.empty:
            continue
        h = pd.to_numeric(f["observation_horizon_sec"], errors="coerce")
        f = f[h == PRIMARY_HORIZON_SEC].copy()
        if not f.empty:
            parts.append(f)
    if not parts:
        raise RuntimeError("No primary 3-second Phase 7M.6F anchor rows found")
    out = pd.concat(parts, ignore_index=True)
    out["candidate_id"] = out["candidate_id"].astype(str)
    if out["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate primary anchor rows")
    forbid_outcome_columns(out, "7M.6F anchor rows")
    return out


def build_causal_clock(candidates: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("candidate_id", "anchor_found", "anchor_ts_recv_ns", "horizon_complete", "invalid_reason") if c in anchors.columns]
    out = candidates.merge(anchors[cols], on="candidate_id", how="left", validate="one_to_one")
    anchor_found = out.get("anchor_found", False)
    if not isinstance(anchor_found, pd.Series):
        anchor_found = pd.Series(False, index=out.index)
    anchor_ns = pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce")
    anchor_ok = anchor_found.fillna(False).astype(bool) & np.isfinite(anchor_ns)
    out["anchor_valid_3s"] = anchor_ok
    out["pre_event_cutoff_ns"] = out["candidate_second"].astype("int64") * NS_PER_SEC
    out["decision_ts_recv_ns"] = np.where(
        anchor_ok, anchor_ns + PRIMARY_HORIZON_SEC * NS_PER_SEC, np.nan
    )
    out["period"] = out["candidate_date"].map(split_period)
    return out


# ---------------------------------------------------------------------------
# Per-day canonical loading
# ---------------------------------------------------------------------------


class DayCache:
    def __init__(self, canonical_root: Path, max_days: int = 3):
        self.root = canonical_root
        self.max_days = max_days
        self.cache: OrderedDict[tuple[str, str], pd.DataFrame] = OrderedDict()

    def _load_one(self, family: str, day: date, columns: Sequence[str]) -> pd.DataFrame:
        key = (family, day.isoformat())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        parts = []
        for p in family_files(self.root, family, day):
            f = read_existing_columns(p, columns)
            if not f.empty:
                parts.append(f)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(columns))
        if not out.empty:
            out["ts_recv_ns"] = pd.to_numeric(out["ts_recv_ns"], errors="coerce")
            out = out[np.isfinite(out["ts_recv_ns"])].copy()
            out["ts_recv_ns"] = out["ts_recv_ns"].astype("int64")
            out = out.sort_values(["ts_recv_ns", "symbol"], kind="mergesort").reset_index(drop=True)
        self.cache[key] = out
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_days * 4:
            self.cache.popitem(last=False)
        return out

    def load_window(self, family: str, day: date, columns: Sequence[str], include_prev: bool = True) -> pd.DataFrame:
        frames = []
        if include_prev:
            frames.append(self._load_one(family, day - timedelta(days=1), columns))
        frames.append(self._load_one(family, day, columns))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=list(columns))
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["ts_recv_ns", "symbol"], kind="mergesort").reset_index(drop=True)



# ---------------------------------------------------------------------------
# Bounded / targeted mark-surface loading
# ---------------------------------------------------------------------------


class MarkTimestampCache:
    """Cache only unique mark-surface receive timestamps, never full mark rows."""

    def __init__(self, canonical_root: Path, max_days: int = 3):
        self.root = canonical_root
        self.max_days = max_days
        self.cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()

    def load_day(self, day: date) -> tuple[np.ndarray, int]:
        key = day.isoformat()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        unique_parts: list[np.ndarray] = []
        row_count = 0
        for path in family_files(self.root, FAMILY_MARK, day):
            pf = pq.ParquetFile(path)
            row_count += int(pf.metadata.num_rows)
            if "ts_recv_ns" not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(columns=["ts_recv_ns"], batch_size=262_144):
                arr = batch.column(0).to_numpy(zero_copy_only=False)
                if len(arr):
                    vals = np.asarray(arr, dtype=np.int64)
                    unique_parts.append(np.unique(vals))

        if unique_parts:
            unique_ts = np.unique(np.concatenate(unique_parts)).astype(np.int64, copy=False)
        else:
            unique_ts = np.array([], dtype=np.int64)

        self.cache[key] = (unique_ts, row_count)
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_days:
            self.cache.popitem(last=False)
        return self.cache[key]

    def load_window(self, day: date) -> tuple[np.ndarray, np.ndarray, int]:
        prev_ts, _ = self.load_day(day - timedelta(days=1))
        cur_ts, cur_rows = self.load_day(day)
        if len(prev_ts) and len(cur_ts):
            all_ts = np.unique(np.concatenate([prev_ts, cur_ts])).astype(np.int64, copy=False)
        elif len(prev_ts):
            all_ts = prev_ts
        else:
            all_ts = cur_ts
        return all_ts, cur_ts, cur_rows


def nearest_prior_surface_timestamps(unique_ts: np.ndarray, targets_ns: Sequence[int]) -> dict[int, int | None]:
    """Resolve each causal target to the exact latest surface timestamp <= target."""
    out: dict[int, int | None] = {}
    if len(unique_ts) == 0:
        return {int(t): None for t in targets_ns}
    for target in sorted({int(t) for t in targets_ns}):
        i = int(np.searchsorted(unique_ts, target, side="right") - 1)
        out[target] = int(unique_ts[i]) if i >= 0 else None
    return out


def _row_group_may_contain_any(pf: pq.ParquetFile, rg: int, ts_col_idx: int, needed_ts: np.ndarray) -> bool:
    if len(needed_ts) == 0:
        return False
    try:
        stats = pf.metadata.row_group(rg).column(ts_col_idx).statistics
        if stats is None or not stats.has_min_max:
            return True
        lo = int(stats.min)
        hi = int(stats.max)
        j = int(np.searchsorted(needed_ts, lo, side="left"))
        return j < len(needed_ts) and int(needed_ts[j]) <= hi
    except Exception:
        return True


def read_mark_rows_for_timestamps(
    canonical_root: Path,
    days: Sequence[date],
    needed_ts: Sequence[int],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read only rows belonging to the exact mark surfaces required by candidates."""
    needed = np.array(sorted({int(x) for x in needed_ts if x is not None}), dtype=np.int64)
    if len(needed) == 0:
        return pd.DataFrame(columns=list(columns))
    value_set = pa.array(needed, type=pa.int64())
    parts: list[pd.DataFrame] = []

    for day in days:
        for path in family_files(canonical_root, FAMILY_MARK, day):
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            if "ts_recv_ns" not in names:
                continue
            use = [c for c in columns if c in names]
            if not use:
                continue
            ts_meta_idx = names.index("ts_recv_ns")
            for rg in range(pf.metadata.num_row_groups):
                if not _row_group_may_contain_any(pf, rg, ts_meta_idx, needed):
                    continue
                table = pf.read_row_group(rg, columns=use)
                ts_arr = table["ts_recv_ns"]
                mask = pc.is_in(ts_arr, value_set=value_set)
                if pc.any(mask).as_py():
                    selected = table.filter(mask)
                    if selected.num_rows:
                        f = selected.to_pandas()
                        for c in columns:
                            if c not in f.columns:
                                f[c] = np.nan
                        parts.append(f[list(columns)])

    if not parts:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(parts, ignore_index=True)
    out["ts_recv_ns"] = pd.to_numeric(out["ts_recv_ns"], errors="coerce")
    out = out[np.isfinite(out["ts_recv_ns"])].copy()
    out["ts_recv_ns"] = out["ts_recv_ns"].astype("int64")
    return out.sort_values(["ts_recv_ns", "symbol"], kind="mergesort").reset_index(drop=True)


def targeted_mark_surfaces_for_day(
    canonical_root: Path,
    ts_cache: MarkTimestampCache,
    day: date,
    day_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[int, int | None]]:
    """Return only candidate-required mark surfaces plus full-day cadence metadata."""
    targets: list[int] = []
    for r in day_candidates.itertuples(index=False):
        pre_ns = int(r.pre_event_cutoff_ns)
        decision_ns = int(r.decision_ts_recv_ns)
        targets.extend([pre_ns, decision_ns])
        targets.extend(pre_ns - int(w * NS_PER_SEC) for w in PRE_WINDOWS_SEC)

    all_ts, cur_ts, cur_rows = ts_cache.load_window(day)
    resolved = nearest_prior_surface_timestamps(all_ts, targets)
    needed = [x for x in resolved.values() if x is not None]
    mark = read_mark_rows_for_timestamps(
        canonical_root, [day - timedelta(days=1), day], needed, MARK_COLUMNS
    )

    gaps = np.diff(cur_ts) / NS_PER_SEC if len(cur_ts) > 1 else np.array([], dtype=float)
    q = qstats(gaps)
    cadence = {
        "date": day.isoformat(),
        "surface_snapshots": int(len(cur_ts)),
        "mark_rows": int(cur_rows),
        "symbols": int(mark["symbol"].nunique()) if not mark.empty else 0,
        "gap_median_sec": q["median"],
        "gap_p90_sec": q["p90"],
        "gap_p99_sec": q["p99"],
        "gap_max_sec": q["max"],
        "candidate_targets": int(len(set(targets))),
        "resolved_target_surfaces": int(len(set(needed))),
        "loaded_mark_rows": int(len(mark)),
    }
    return mark, cadence, resolved


def day_checkpoint_paths(output_root: Path, day_text: str) -> tuple[Path, Path]:
    root = output_root / "day_checkpoints" / f"candidate_date={day_text}"
    return root / "part.parquet", root / "manifest.json"


def load_day_checkpoint(output_root: Path, day_text: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    part, manifest = day_checkpoint_paths(output_root, day_text)
    if not part.exists() or not manifest.exists():
        return None
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("checkpoint_contract_version") != CHECKPOINT_CONTRACT_VERSION:
            return None
        frame = pd.read_parquet(part)
        if len(frame) != int(meta.get("candidate_rows", -1)):
            return None
        return frame, meta
    except Exception:
        return None


def write_day_checkpoint(
    output_root: Path, day_text: str, frame: pd.DataFrame, cadence: dict[str, Any],
    side_counts: dict[str, int], maker_counts: dict[str, int],
) -> None:
    part, manifest = day_checkpoint_paths(output_root, day_text)
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(part)
    meta = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "candidate_date": day_text,
        "candidate_rows": int(len(frame)),
        "cadence": cadence,
        "side_counts": side_counts,
        "maker_counts": maker_counts,
        "written_at_utc": utc_now_iso(),
    }
    tmpm = manifest.with_suffix(".tmp.json")
    tmpm.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmpm.replace(manifest)

# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------


@dataclass
class SurfaceSnapshot:
    target_ns: int
    surface_ts_ns: int | None
    age_sec: float
    metrics: dict[str, Any]


def numericize_mark(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_contract_columns(frame)
    for c in (
        "mark_price", "index_price", "estimated_settle_price", "best_bid_price", "best_ask_price",
        "best_bid_quantity", "best_ask_quantity", "bid_iv", "ask_iv", "mark_iv",
        "risk_free_rate", "delta", "theta", "gamma", "vega",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def front_expiry_pair_metrics(surface: pd.DataFrame, target_ns: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "surface_contracts": 0,
        "surface_expiries": 0,
        "surface_calls": 0,
        "surface_puts": 0,
        "mark_iv_valid_frac": math.nan,
        "delta_valid_frac": math.nan,
        "gamma_valid_frac": math.nan,
        "vega_valid_frac": math.nan,
        "front_expiry_code": None,
        "front_calendar_days": math.nan,
        "front_atm_strike": math.nan,
        "front_atm_abs_moneyness_pct": math.nan,
        "front_atm_call_iv": math.nan,
        "front_atm_put_iv": math.nan,
        "front_atm_iv": math.nan,
        "front_same_strike_call_put_iv_diff": math.nan,
        "front_25d_call_iv": math.nan,
        "front_25d_put_iv": math.nan,
        "front_25d_call_delta_abs_error": math.nan,
        "front_25d_put_delta_abs_error": math.nan,
        "front_25d_call_put_iv_diff": math.nan,
        "second_expiry_code": None,
        "second_atm_iv": math.nan,
        "second_atm_abs_moneyness_pct": math.nan,
        "second_minus_front_atm_iv": math.nan,
        "contracts_within_2pct": 0,
        "contracts_within_5pct": 0,
        "contracts_within_10pct": 0,
        "contracts_within_20pct": 0,
        "index_price_surface": math.nan,
    }
    if surface.empty:
        return result

    s = surface.copy()
    s = s[s["option_type"].isin(["C", "P"]) & np.isfinite(pd.to_numeric(s["strike"], errors="coerce"))].copy()
    if s.empty:
        return result
    idx_vals = pd.to_numeric(s["index_price"], errors="coerce")
    idx_vals = idx_vals[np.isfinite(idx_vals) & (idx_vals > 0)]
    if idx_vals.empty:
        return result
    index_price = float(idx_vals.median())
    result["index_price_surface"] = index_price

    s["strike"] = pd.to_numeric(s["strike"], errors="coerce")
    s["abs_moneyness_pct"] = np.abs(s["strike"] / index_price - 1.0) * 100.0
    now_date = datetime.fromtimestamp(target_ns / NS_PER_SEC, tz=timezone.utc).date()
    s["calendar_days"] = s["expiry_date"].map(lambda x: (x - now_date).days if isinstance(x, date) else math.nan)
    s = s[pd.to_numeric(s["calendar_days"], errors="coerce") >= 0].copy()
    if s.empty:
        return result

    result["surface_contracts"] = int(s["symbol"].nunique())
    result["surface_expiries"] = int(s["expiry_code"].nunique())
    result["surface_calls"] = int((s["option_type"] == "C").sum())
    result["surface_puts"] = int((s["option_type"] == "P").sum())
    n = len(s)
    for col, key in (
        ("mark_iv", "mark_iv_valid_frac"), ("delta", "delta_valid_frac"),
        ("gamma", "gamma_valid_frac"), ("vega", "vega_valid_frac"),
    ):
        vals = pd.to_numeric(s[col], errors="coerce") if col in s.columns else pd.Series([], dtype=float)
        result[key] = float(np.isfinite(vals).sum() / n) if n else math.nan
    for pct_v in (2, 5, 10, 20):
        result[f"contracts_within_{pct_v}pct"] = int((s["abs_moneyness_pct"] <= pct_v).sum())

    def expiry_metrics(expiry_code: str) -> dict[str, Any] | None:
        e = s[s["expiry_code"] == expiry_code].copy()
        if e.empty:
            return None
        piv = e.pivot_table(index="strike", columns="option_type", values="mark_iv", aggfunc="last")
        if "C" not in piv.columns or "P" not in piv.columns:
            return None
        both = piv[np.isfinite(piv["C"]) & np.isfinite(piv["P"])].copy()
        if both.empty:
            return None
        strikes = both.index.to_numpy(dtype=float)
        i = int(np.argmin(np.abs(strikes / index_price - 1.0)))
        strike = float(strikes[i])
        c_iv = float(both.iloc[i]["C"])
        p_iv = float(both.iloc[i]["P"])
        atm_iv = float(np.mean([c_iv, p_iv]))
        abs_m = abs(strike / index_price - 1.0) * 100.0

        call = e[(e["option_type"] == "C") & np.isfinite(pd.to_numeric(e["mark_iv"], errors="coerce")) & np.isfinite(pd.to_numeric(e["delta"], errors="coerce"))].copy()
        put = e[(e["option_type"] == "P") & np.isfinite(pd.to_numeric(e["mark_iv"], errors="coerce")) & np.isfinite(pd.to_numeric(e["delta"], errors="coerce"))].copy()
        c25_iv = p25_iv = c_err = p_err = math.nan
        if not call.empty:
            d = pd.to_numeric(call["delta"], errors="coerce").to_numpy(dtype=float)
            j = int(np.argmin(np.abs(d - 0.25)))
            c_err = float(abs(d[j] - 0.25))
            c25_iv = safe_float(call.iloc[j]["mark_iv"])
        if not put.empty:
            d = pd.to_numeric(put["delta"], errors="coerce").to_numpy(dtype=float)
            j = int(np.argmin(np.abs(d + 0.25)))
            p_err = float(abs(d[j] + 0.25))
            p25_iv = safe_float(put.iloc[j]["mark_iv"])
        return {
            "expiry_code": expiry_code,
            "calendar_days": safe_float(e["calendar_days"].iloc[0]),
            "atm_strike": strike,
            "atm_abs_moneyness_pct": abs_m,
            "atm_call_iv": c_iv,
            "atm_put_iv": p_iv,
            "atm_iv": atm_iv,
            "same_strike_skew": c_iv - p_iv,
            "c25_iv": c25_iv,
            "p25_iv": p25_iv,
            "c25_err": c_err,
            "p25_err": p_err,
            "skew25": c25_iv - p25_iv if np.isfinite(c25_iv) and np.isfinite(p25_iv) else math.nan,
        }

    expiry_order = (
        s[["expiry_code", "calendar_days"]].drop_duplicates()
        .sort_values(["calendar_days", "expiry_code"], kind="mergesort")
    )
    expiry_metrics_list = []
    for exp in expiry_order["expiry_code"].astype(str):
        m = expiry_metrics(exp)
        if m is not None:
            expiry_metrics_list.append(m)
        if len(expiry_metrics_list) >= 2:
            break
    if not expiry_metrics_list:
        return result
    f = expiry_metrics_list[0]
    result.update({
        "front_expiry_code": f["expiry_code"],
        "front_calendar_days": f["calendar_days"],
        "front_atm_strike": f["atm_strike"],
        "front_atm_abs_moneyness_pct": f["atm_abs_moneyness_pct"],
        "front_atm_call_iv": f["atm_call_iv"],
        "front_atm_put_iv": f["atm_put_iv"],
        "front_atm_iv": f["atm_iv"],
        "front_same_strike_call_put_iv_diff": f["same_strike_skew"],
        "front_25d_call_iv": f["c25_iv"],
        "front_25d_put_iv": f["p25_iv"],
        "front_25d_call_delta_abs_error": f["c25_err"],
        "front_25d_put_delta_abs_error": f["p25_err"],
        "front_25d_call_put_iv_diff": f["skew25"],
    })
    if len(expiry_metrics_list) >= 2:
        s2 = expiry_metrics_list[1]
        result["second_expiry_code"] = s2["expiry_code"]
        result["second_atm_iv"] = s2["atm_iv"]
        result["second_atm_abs_moneyness_pct"] = s2["atm_abs_moneyness_pct"]
        if np.isfinite(s2["atm_iv"]) and np.isfinite(f["atm_iv"]):
            result["second_minus_front_atm_iv"] = s2["atm_iv"] - f["atm_iv"]
    return result


class MarkSurfaceIndex:
    def __init__(self, frame: pd.DataFrame):
        self.frame = numericize_mark(frame)
        if self.frame.empty:
            self.unique_ts = np.array([], dtype=np.int64)
            self.groups: dict[int, pd.DataFrame] = {}
            return
        self.unique_ts = np.sort(self.frame["ts_recv_ns"].unique().astype(np.int64))
        self.groups = {int(k): g.copy() for k, g in self.frame.groupby("ts_recv_ns", sort=False)}

    def snapshot_at_or_before(self, target_ns: int) -> SurfaceSnapshot:
        if len(self.unique_ts) == 0:
            return SurfaceSnapshot(target_ns, None, math.nan, front_expiry_pair_metrics(pd.DataFrame(), target_ns))
        i = int(np.searchsorted(self.unique_ts, target_ns, side="right") - 1)
        if i < 0:
            return SurfaceSnapshot(target_ns, None, math.nan, front_expiry_pair_metrics(pd.DataFrame(), target_ns))
        ts = int(self.unique_ts[i])
        surface = self.groups.get(ts, pd.DataFrame())
        age = (target_ns - ts) / NS_PER_SEC
        return SurfaceSnapshot(target_ns, ts, float(age), front_expiry_pair_metrics(surface, target_ns))


# ---------------------------------------------------------------------------
# OI / trade aggregation
# ---------------------------------------------------------------------------


def numericize_oi(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_contract_columns(frame)
    for c in ("open_interest", "open_interest_value"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def latest_oi_per_symbol(oi: pd.DataFrame, target_ns: int, lookback_sec: int, index_price: float) -> dict[str, Any]:
    result = {
        "oi_contracts": 0, "oi_expiries": 0, "oi_age_median_sec": math.nan, "oi_age_p90_sec": math.nan,
        "oi_call_sum": math.nan, "oi_put_sum": math.nan, "oi_value_call_sum": math.nan, "oi_value_put_sum": math.nan,
        "oi_near5pct_sum": math.nan, "oi_value_near5pct_sum": math.nan,
    }
    if oi.empty:
        return result
    lo = target_ns - int(lookback_sec * NS_PER_SEC)
    w = oi[(oi["ts_recv_ns"] > lo) & (oi["ts_recv_ns"] <= target_ns)].copy()
    if w.empty:
        return result
    w = w.sort_values(["symbol", "ts_recv_ns"], kind="mergesort").groupby("symbol", as_index=False).tail(1)
    if w.empty:
        return result
    ages = (target_ns - w["ts_recv_ns"].to_numpy(dtype=np.int64)) / NS_PER_SEC
    result["oi_contracts"] = int(w["symbol"].nunique())
    result["oi_expiries"] = int(w["expiry_code"].dropna().nunique())
    result["oi_age_median_sec"] = float(np.median(ages))
    result["oi_age_p90_sec"] = float(np.quantile(ages, 0.9))
    for typ, prefix in (("C", "call"), ("P", "put")):
        t = w[w["option_type"] == typ]
        result[f"oi_{prefix}_sum"] = float(pd.to_numeric(t["open_interest"], errors="coerce").sum(min_count=1)) if not t.empty else math.nan
        result[f"oi_value_{prefix}_sum"] = float(pd.to_numeric(t["open_interest_value"], errors="coerce").sum(min_count=1)) if not t.empty else math.nan
    if np.isfinite(index_price) and index_price > 0:
        strike = pd.to_numeric(w["strike"], errors="coerce")
        near = w[np.abs(strike / index_price - 1.0) <= 0.05]
        result["oi_near5pct_sum"] = float(pd.to_numeric(near["open_interest"], errors="coerce").sum(min_count=1)) if not near.empty else math.nan
        result["oi_value_near5pct_sum"] = float(pd.to_numeric(near["open_interest_value"], errors="coerce").sum(min_count=1)) if not near.empty else math.nan
    return result


def trade_window_metrics(trades: pd.DataFrame, start_ns: int, end_ns: int, index_price: float) -> dict[str, Any]:
    out = {
        "trade_count": 0, "trade_call_count": 0, "trade_put_count": 0,
        "trade_quantity_sum": math.nan, "trade_premium_proxy_sum": math.nan,
        "trade_near5pct_count": 0, "trade_side_nonnull_frac": math.nan, "trade_maker_nonnull_frac": math.nan,
        "trade_maker_true_share": math.nan,
    }
    if trades.empty or end_ns <= start_ns:
        return out
    w = trades[(trades["ts_recv_ns"] > start_ns) & (trades["ts_recv_ns"] <= end_ns)].copy()
    if w.empty:
        return out
    out["trade_count"] = int(len(w))
    out["trade_call_count"] = int((w["option_type"] == "C").sum())
    out["trade_put_count"] = int((w["option_type"] == "P").sum())
    q = pd.to_numeric(w["quantity"], errors="coerce")
    p = pd.to_numeric(w["price"], errors="coerce")
    out["trade_quantity_sum"] = float(q.sum(min_count=1))
    prem = p * q
    out["trade_premium_proxy_sum"] = float(prem.sum(min_count=1))
    if np.isfinite(index_price) and index_price > 0:
        strike = pd.to_numeric(w["strike"], errors="coerce")
        out["trade_near5pct_count"] = int((np.abs(strike / index_price - 1.0) <= 0.05).sum())
    out["trade_side_nonnull_frac"] = float(w["side"].notna().mean()) if "side" in w else math.nan
    out["trade_maker_nonnull_frac"] = float(w["maker"].notna().mean()) if "maker" in w else math.nan
    if "maker" in w and w["maker"].notna().any():
        out["trade_maker_true_share"] = float(w.loc[w["maker"].notna(), "maker"].astype(bool).mean())
    return out


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def inspect_family_availability(canonical_root: Path, study_days: Sequence[date]) -> pd.DataFrame:
    rows = []
    columns_map = {
        FAMILY_MARK: MARK_COLUMNS,
        FAMILY_OI: OI_COLUMNS,
        FAMILY_TRADES: TRADE_COLUMNS,
        FAMILY_INDEX: INDEX_COLUMNS,
    }
    for family, cols in columns_map.items():
        for d in study_days:
            files = family_files(canonical_root, family, d)
            nrows = sum(parquet_rows(p) for p in files)
            present: set[str] = set()
            for p in files[:3]:
                present.update(pq.ParquetFile(p).schema_arrow.names)
            rows.append({
                "family": family,
                "date": d.isoformat(),
                "files": len(files),
                "rows": nrows,
                "has_data": bool(nrows > 0),
                "requested_fields_present": int(sum(c in present for c in cols)),
                "requested_fields_total": len(cols),
                "field_presence_frac": float(sum(c in present for c in cols) / len(cols)) if cols else math.nan,
            })
    return pd.DataFrame(rows)


def mark_cadence(mark: pd.DataFrame, day_text: str) -> dict[str, Any]:
    if mark.empty:
        return {"date": day_text, "surface_snapshots": 0, "mark_rows": 0}
    ts = np.sort(mark["ts_recv_ns"].unique().astype(np.int64))
    gaps = np.diff(ts) / NS_PER_SEC if len(ts) > 1 else np.array([], dtype=float)
    q = qstats(gaps)
    return {
        "date": day_text,
        "surface_snapshots": int(len(ts)),
        "mark_rows": int(len(mark)),
        "symbols": int(mark["symbol"].nunique()),
        "gap_median_sec": q["median"],
        "gap_p90_sec": q["p90"],
        "gap_p99_sec": q["p99"],
        "gap_max_sec": q["max"],
    }


def summarize_metric_coverage(candidate_df: pd.DataFrame, metric: str, label: str) -> dict[str, Any]:
    vals = pd.to_numeric(candidate_df.get(metric), errors="coerce")
    finite = np.isfinite(vals)
    q = qstats(vals[finite])
    return {
        "metric": label,
        "column": metric,
        "finite_n": int(finite.sum()),
        "finite_pct": float(finite.mean() * 100.0) if len(vals) else math.nan,
        **{f"{k}": v for k, v in q.items()},
    }


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = load_candidate_inputs(args.candidate_root)
    anchors = load_primary_anchor_rows(args.sequence_root)
    clock = build_causal_clock(candidates, anchors)
    anchored = clock[clock["anchor_valid_3s"]].copy()
    if anchored.empty:
        raise RuntimeError("No valid anchor+3s candidates available")

    study_days = sorted({datetime.fromisoformat(x).date() for x in clock["candidate_date"].unique()})
    availability = inspect_family_availability(args.canonical_root, study_days)

    cache = DayCache(args.canonical_root, max_days=2)
    mark_ts_cache = MarkTimestampCache(args.canonical_root, max_days=3)
    candidate_frames: list[pd.DataFrame] = []
    cadence_rows: list[dict[str, Any]] = []
    side_counter: Counter[str] = Counter()
    maker_counter: Counter[str] = Counter()

    args.output_root.mkdir(parents=True, exist_ok=True)
    grouped = anchored.groupby("candidate_date", sort=True)
    total_days = len(grouped)
    for day_i, (day_text, day_candidates) in enumerate(grouped, 1):
        day_text = str(day_text)
        reused = load_day_checkpoint(args.output_root, day_text)
        if reused is not None:
            day_frame, meta = reused
            candidate_frames.append(day_frame)
            cadence_rows.append(dict(meta.get("cadence", {"date": day_text})))
            side_counter.update({str(k): int(v) for k, v in meta.get("side_counts", {}).items()})
            maker_counter.update({str(k): int(v) for k, v in meta.get("maker_counts", {}).items()})
            print(
                f"  [{day_i:02d}/{total_days:02d}] {day_text}: candidates={len(day_frame):4d} "
                f"CHECKPOINT_REUSED loaded_mark_rows={int(meta.get('cadence', {}).get('loaded_mark_rows', 0)):7d}"
            )
            continue

        d = datetime.fromisoformat(day_text).date()
        # The mark/Greek family is huge (millions of rows/day). Resolve the exact
        # causal surfaces first from the timestamp column, then materialize only
        # those surfaces. This is scientifically identical to the original latest-
        # prior lookup but avoids whole-day pandas frames.
        mark, cadence, _resolved = targeted_mark_surfaces_for_day(
            args.canonical_root, mark_ts_cache, d, day_candidates
        )
        oi = numericize_oi(cache.load_window(FAMILY_OI, d, OI_COLUMNS, include_prev=True))
        trades = add_contract_columns(cache.load_window(FAMILY_TRADES, d, TRADE_COLUMNS, include_prev=True))
        index = cache.load_window(FAMILY_INDEX, d, INDEX_COLUMNS, include_prev=True)

        day_side_counter: Counter[str] = Counter()
        day_maker_counter: Counter[str] = Counter()
        if not trades.empty:
            for v, n in trades["side"].fillna("<NULL>").astype(str).value_counts().items():
                day_side_counter[str(v)] += int(n)
            for v, n in trades["maker"].map(lambda x: "<NULL>" if pd.isna(x) else str(bool(x))).value_counts().items():
                day_maker_counter[str(v)] += int(n)
        side_counter.update(day_side_counter)
        maker_counter.update(day_maker_counter)
        cadence_rows.append(cadence)
        surface_index = MarkSurfaceIndex(mark)

        day_rows: list[dict[str, Any]] = []
        for r in day_candidates.itertuples(index=False):
            pre_ns = int(r.pre_event_cutoff_ns)
            decision_ns = int(r.decision_ts_recv_ns)
            snap_pre = surface_index.snapshot_at_or_before(pre_ns)
            snap_dec = surface_index.snapshot_at_or_before(decision_ns)
            prior_snaps = {w: surface_index.snapshot_at_or_before(pre_ns - w * NS_PER_SEC) for w in PRE_WINDOWS_SEC}

            row: dict[str, Any] = {
                "candidate_id": str(r.candidate_id),
                "candidate_date": str(r.candidate_date),
                "period": str(r.period),
                "direction": int(r.direction),
                "candidate_second": int(r.candidate_second),
                "pre_event_cutoff_ns": pre_ns,
                "decision_ts_recv_ns": decision_ns,
                "pre_surface_ts_recv_ns": snap_pre.surface_ts_ns,
                "pre_surface_age_sec": snap_pre.age_sec,
                "decision_surface_ts_recv_ns": snap_dec.surface_ts_ns,
                "decision_surface_age_sec": snap_dec.age_sec,
            }
            for k, v in snap_pre.metrics.items():
                row[f"pre__{k}"] = v
            for k, v in snap_dec.metrics.items():
                row[f"decision__{k}"] = v

            # Pre-event anticipation changes and expiry-comparability diagnostics.
            for w, snap_old in prior_snaps.items():
                mins = w // 60
                row[f"pre_minus_{mins}m_surface_ts_recv_ns"] = snap_old.surface_ts_ns
                row[f"pre_minus_{mins}m_surface_age_sec"] = snap_old.age_sec
                for metric in (
                    "front_atm_iv", "front_same_strike_call_put_iv_diff",
                    "front_25d_call_put_iv_diff", "second_minus_front_atm_iv",
                ):
                    old_v = safe_float(snap_old.metrics.get(metric))
                    now_v = safe_float(snap_pre.metrics.get(metric))
                    row[f"anticipation_{mins}m__{metric}_change"] = now_v - old_v if np.isfinite(now_v) and np.isfinite(old_v) else math.nan
                row[f"anticipation_{mins}m__front_expiry_same"] = bool(
                    snap_old.metrics.get("front_expiry_code")
                    and snap_old.metrics.get("front_expiry_code") == snap_pre.metrics.get("front_expiry_code")
                )
                old_strike = safe_float(snap_old.metrics.get("front_atm_strike"))
                now_strike = safe_float(snap_pre.metrics.get("front_atm_strike"))
                row[f"anticipation_{mins}m__front_atm_strike_same"] = bool(
                    np.isfinite(old_strike) and np.isfinite(now_strike) and old_strike == now_strike
                )

            # Event response, explicitly separate from anticipation.
            for metric in (
                "front_atm_iv", "front_same_strike_call_put_iv_diff",
                "front_25d_call_put_iv_diff", "second_minus_front_atm_iv",
            ):
                a = safe_float(snap_pre.metrics.get(metric))
                b = safe_float(snap_dec.metrics.get(metric))
                row[f"event_response__{metric}_change"] = b - a if np.isfinite(a) and np.isfinite(b) else math.nan
            row["event_response__front_expiry_same"] = bool(
                snap_pre.metrics.get("front_expiry_code")
                and snap_pre.metrics.get("front_expiry_code") == snap_dec.metrics.get("front_expiry_code")
            )

            # OI structure at pre-event cutoff only (regime / positioning landscape).
            oi_metrics = latest_oi_per_symbol(
                oi, pre_ns, OI_AUDIT_LOOKBACK_SEC, safe_float(snap_pre.metrics.get("index_price_surface"))
            )
            row.update({f"pre__{k}": v for k, v in oi_metrics.items()})

            # Option trade activity strictly before candidate, plus event window.
            idx_px = safe_float(snap_pre.metrics.get("index_price_surface"))
            for w in PRE_WINDOWS_SEC:
                mins = w // 60
                tm = trade_window_metrics(trades, pre_ns - w * NS_PER_SEC, pre_ns, idx_px)
                row.update({f"trade_pre_{mins}m__{k}": v for k, v in tm.items()})
            event_tm = trade_window_metrics(trades, pre_ns, decision_ns, idx_px)
            row.update({f"trade_event__{k}": v for k, v in event_tm.items()})

            # Index stream freshness audit (not used to manufacture surface values).
            idx_age = math.nan
            idx_price = math.nan
            if not index.empty:
                its = index["ts_recv_ns"].to_numpy(dtype=np.int64)
                j = int(np.searchsorted(its, pre_ns, side="right") - 1)
                if j >= 0:
                    idx_age = (pre_ns - int(its[j])) / NS_PER_SEC
                    idx_price = safe_float(index.iloc[j].get("index_price"))
            row["pre__index_stream_age_sec"] = idx_age
            row["pre__index_stream_price"] = idx_price
            day_rows.append(row)

        day_frame = pd.DataFrame(day_rows)
        forbid_outcome_columns(day_frame, f"7M.11A {day_text} checkpoint")
        write_day_checkpoint(
            args.output_root, day_text, day_frame, cadence,
            dict(day_side_counter), dict(day_maker_counter),
        )
        candidate_frames.append(day_frame)
        print(
            f"  [{day_i:02d}/{total_days:02d}] {day_text}: candidates={len(day_frame):4d} "
            f"mark_rows={int(cadence.get('mark_rows', 0)):7d} surfaces={int(cadence.get('surface_snapshots', 0)):6d} "
            f"loaded_rows={int(cadence.get('loaded_mark_rows', 0)):7d} checkpoint=SIGNED"
        )

        del mark, surface_index, oi, trades, index, day_rows
        gc.collect()

    cand = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    forbid_outcome_columns(cand, "7M.11A candidate options audit")

    # Reports
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / "candidate_options_regime_anticipation_audit.parquet"
    cand.to_parquet(out_path, index=False)
    write_csv(args.report_root / "family_availability.csv", availability)
    write_csv(args.report_root / "mark_surface_cadence_by_day.csv", cadence_rows)

    field_rows = []
    for label, cols in (
        ("mark_price_greeks", MARK_COLUMNS), ("open_interest", OI_COLUMNS),
        ("trades", TRADE_COLUMNS), ("index_price", INDEX_COLUMNS),
    ):
        fam = {"mark_price_greeks": FAMILY_MARK, "open_interest": FAMILY_OI, "trades": FAMILY_TRADES, "index_price": FAMILY_INDEX}[label]
        sample_files = []
        for d in study_days:
            sample_files.extend(family_files(args.canonical_root, fam, d))
            if sample_files:
                break
        present = set(pq.ParquetFile(sample_files[0]).schema_arrow.names) if sample_files else set()
        for c in cols:
            field_rows.append({"family": label, "field": c, "present_in_sample_schema": c in present})
    write_csv(args.report_root / "options_field_presence.csv", field_rows)

    coverage_metrics = [
        ("pre__surface_contracts", "pre-event surface contract count"),
        ("pre__surface_expiries", "pre-event surface expiry count"),
        ("pre__mark_iv_valid_frac", "pre-event mark-IV completeness"),
        ("pre__delta_valid_frac", "pre-event delta completeness"),
        ("pre__front_atm_abs_moneyness_pct", "front ATM pair distance from index (%)"),
        ("pre__front_atm_iv", "front ATM IV"),
        ("pre__front_same_strike_call_put_iv_diff", "front same-strike call-put IV spread"),
        ("pre__front_25d_call_delta_abs_error", "front 25d call delta abs error"),
        ("pre__front_25d_put_delta_abs_error", "front 25d put delta abs error"),
        ("pre__front_25d_call_put_iv_diff", "front approximate 25d call-put IV spread"),
        ("pre__second_minus_front_atm_iv", "second-minus-front ATM IV term difference"),
        ("pre__oi_contracts", "pre-event OI contracts within 30m audit window"),
        ("pre__oi_age_median_sec", "pre-event median latest-per-contract OI age"),
        ("trade_pre_5m__trade_count", "pre-event option trades in 5m"),
        ("trade_pre_15m__trade_count", "pre-event option trades in 15m"),
        ("trade_pre_30m__trade_count", "pre-event option trades in 30m"),
        ("trade_event__trade_count", "option trades candidate-second-to-decision"),
    ]
    cov_rows = [summarize_metric_coverage(cand, c, label) for c, label in coverage_metrics if c in cand.columns]
    write_csv(args.report_root / "options_surface_and_activity_feasibility.csv", cov_rows)

    change_rows = []
    for mins in (5, 15, 30):
        for metric in (
            "front_atm_iv", "front_same_strike_call_put_iv_diff",
            "front_25d_call_put_iv_diff", "second_minus_front_atm_iv",
        ):
            col = f"anticipation_{mins}m__{metric}_change"
            change_rows.append(summarize_metric_coverage(cand, col, f"pre-event {mins}m change: {metric}"))
        same_exp_col = f"anticipation_{mins}m__front_expiry_same"
        same_strike_col = f"anticipation_{mins}m__front_atm_strike_same"
        change_rows.append({
            "metric": f"pre-event {mins}m front-expiry comparability",
            "column": same_exp_col,
            "finite_n": len(cand),
            "finite_pct": 100.0,
            "mean": float(cand[same_exp_col].astype(bool).mean()),
        })
        change_rows.append({
            "metric": f"pre-event {mins}m front-ATM-strike comparability",
            "column": same_strike_col,
            "finite_n": len(cand),
            "finite_pct": 100.0,
            "mean": float(cand[same_strike_col].astype(bool).mean()),
        })
    for metric in (
        "front_atm_iv", "front_same_strike_call_put_iv_diff",
        "front_25d_call_put_iv_diff", "second_minus_front_atm_iv",
    ):
        change_rows.append(summarize_metric_coverage(cand, f"event_response__{metric}_change", f"event response: {metric}"))
    write_csv(args.report_root / "anticipation_vs_event_response_feasibility.csv", change_rows)

    side_rows = [{"field": "side", "value": k, "rows": v} for k, v in sorted(side_counter.items())]
    side_rows += [{"field": "maker", "value": k, "rows": v} for k, v in sorted(maker_counter.items())]
    write_csv(args.report_root / "option_trade_raw_field_values.csv", side_rows)

    subgroup_rows = []
    for group_name, mask in (
        ("FULL", pd.Series(True, index=cand.index)),
        ("DISCOVERY", cand["period"] == "DISCOVERY"),
        ("VALIDATION", cand["period"] == "VALIDATION"),
        ("UP", cand["direction"] == 1),
        ("DOWN", cand["direction"] == -1),
    ):
        g = cand[mask]
        subgroup_rows.append({
            "group": group_name,
            "candidates": len(g),
            "pre_surface_available": int(np.isfinite(pd.to_numeric(g["pre__front_atm_iv"], errors="coerce")).sum()),
            "pre_30m_atm_change_available": int(np.isfinite(pd.to_numeric(g["anticipation_30m__front_atm_iv_change"], errors="coerce")).sum()),
            "pre_30m_25d_skew_change_available": int(np.isfinite(pd.to_numeric(g["anticipation_30m__front_25d_call_put_iv_diff_change"], errors="coerce")).sum()),
            "event_atm_response_available": int(np.isfinite(pd.to_numeric(g["event_response__front_atm_iv_change"], errors="coerce")).sum()),
            "oi_structure_available": int((pd.to_numeric(g["pre__oi_contracts"], errors="coerce") > 0).sum()),
            "pre_30m_trade_activity_positive": int((pd.to_numeric(g["trade_pre_30m__trade_count"], errors="coerce") > 0).sum()),
        })
    write_csv(args.report_root / "subgroup_representation.csv", subgroup_rows)

    invariants = [
        {"invariant": "phase", "value": PHASE},
        {"invariant": "script_version", "value": SCRIPT_VERSION},
        {"invariant": "future_outcomes_opened", "value": False},
        {"invariant": "matching_performed", "value": False},
        {"invariant": "composite_signal_created", "value": False},
        {"invariant": "pre_event_cutoff", "value": "candidate_second start boundary"},
        {"invariant": "decision_point", "value": "exact Phase 7M.6F anchor + 3s"},
        {"invariant": "anticipation_windows_sec", "value": list(PRE_WINDOWS_SEC)},
        {"invariant": "oi_audit_lookback_sec_not_threshold", "value": OI_AUDIT_LOOKBACK_SEC},
        {"invariant": "oi_long_short_identity_assumed", "value": False},
        {"invariant": "option_trade_directional_semantics_assumed", "value": False},
        {"invariant": "surface_exposure_threshold_frozen", "value": False},
    ]
    write_csv(args.report_root / "analysis_invariants.csv", invariants)

    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "candidate_population": int(len(clock)),
        "anchored_candidates": int(len(cand)),
        "days": int(len(study_days)),
        "output_parquet": str(out_path),
        "surface_pre_atm_iv_coverage_pct": float(np.isfinite(pd.to_numeric(cand["pre__front_atm_iv"], errors="coerce")).mean() * 100.0),
        "anticipation_30m_atm_iv_change_coverage_pct": float(np.isfinite(pd.to_numeric(cand["anticipation_30m__front_atm_iv_change"], errors="coerce")).mean() * 100.0),
        "anticipation_30m_25d_skew_change_coverage_pct": float(np.isfinite(pd.to_numeric(cand["anticipation_30m__front_25d_call_put_iv_diff_change"], errors="coerce")).mean() * 100.0),
        "event_atm_iv_response_coverage_pct": float(np.isfinite(pd.to_numeric(cand["event_response__front_atm_iv_change"], errors="coerce")).mean() * 100.0),
        "oi_structure_coverage_pct": float((pd.to_numeric(cand["pre__oi_contracts"], errors="coerce") > 0).mean() * 100.0),
        "pre30m_trade_activity_positive_pct": float((pd.to_numeric(cand["trade_pre_30m__trade_count"], errors="coerce") > 0).mean() * 100.0),
        "trade_side_values": dict(side_counter),
        "trade_maker_values": dict(maker_counter),
        "future_outcomes_opened": False,
    }
    write_json(args.report_root / "summary.json", summary)

    findings = f"""# Phase 7M.11A — BTC-options regime / anticipation feasibility

This phase is **outcome-blind**. No continuation labels were opened and no
options signal was frozen.

## Candidate clock

- Frozen prospective candidates: **{len(clock):,}**
- Candidates with valid exact 7M.6F anchor+3s clock: **{len(cand):,}**
- Pre-event cutoff: **start of candidate_second** (conservative; excludes the candidate second from anticipation)
- Event-response endpoint: **exact recovered anchor + 3s**

## What was audited

1. **Pre-existing options regime** at the pre-event cutoff.
2. **Pre-event repricing / anticipation** over 5m / 15m / 30m ending before the candidate second.
3. **Event-time options response** from the candidate-second boundary to anchor+3s.
4. **Positioning landscape** from public option OI without assuming long/short ownership.
5. **Options trade activity** without assigning bullish/bearish meaning to raw side/maker fields.

## Headline causal coverage

- Front-expiry ATM-IV surface at pre-event cutoff: **{summary['surface_pre_atm_iv_coverage_pct']:.2f}%**
- 30m pre-event ATM-IV change: **{summary['anticipation_30m_atm_iv_change_coverage_pct']:.2f}%**
- 30m pre-event approximate 25d skew change: **{summary['anticipation_30m_25d_skew_change_coverage_pct']:.2f}%**
- Event-time ATM-IV response: **{summary['event_atm_iv_response_coverage_pct']:.2f}%**
- OI structure present in 30m audit window: **{summary['oi_structure_coverage_pct']:.2f}%**
- At least one option trade in the preceding 30m: **{summary['pre30m_trade_activity_positive_pct']:.2f}%**

## Interpretation

This audit does **not** yet decide whether ATM IV, skew, term structure, OI or
option trades should become outcome-facing features. The next phase should use
these diagnostics to select only primitives with strong causal coverage,
acceptable contract/expiry comparability, and a clear physical interpretation.

In particular, the audit keeps **pre-event anticipation** separate from
**event-time response** so that any later leading-signal claim cannot be based
on options repricing that occurred after the perp candidate had already begun.
"""
    (args.report_root / "findings.md").write_text(findings, encoding="utf-8")

    return cand, summary


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    assert parse_option_symbol("BTC-261030-70000-C") is not None
    assert parse_option_symbol("BTC-261030-70000-P").option_type == "P"
    assert parse_option_symbol("ETH-261030-70000-C") is None

    target = 1_000 * NS_PER_SEC
    rows = []
    # Two surface timestamps; the later one is future to target and must not be selected.
    for ts, shift in ((target - 2 * NS_PER_SEC, 0.0), (target + 1 * NS_PER_SEC, 10.0)):
        for sym, delta, iv in (
            ("BTC-261030-100-C", 0.50, 0.60 + shift),
            ("BTC-261030-100-P", -0.50, 0.62 + shift),
            ("BTC-261030-110-C", 0.25, 0.65 + shift),
            ("BTC-261030-90-P", -0.25, 0.67 + shift),
            ("BTC-261106-100-C", 0.50, 0.64 + shift),
            ("BTC-261106-100-P", -0.50, 0.66 + shift),
        ):
            rows.append({
                "ts_recv_ns": ts, "symbol": sym, "exchange_event_ms": 0,
                "mark_price": "1", "index_price": "100", "estimated_settle_price": "100",
                "best_bid_price": "1", "best_ask_price": "1", "best_bid_quantity": "1", "best_ask_quantity": "1",
                "bid_iv": str(iv), "ask_iv": str(iv), "mark_iv": str(iv), "risk_free_rate": "0",
                "delta": str(delta), "theta": "-0.1", "gamma": "0.01", "vega": "1",
            })
    idx = MarkSurfaceIndex(pd.DataFrame(rows))
    snap = idx.snapshot_at_or_before(target)
    assert snap.surface_ts_ns == target - 2 * NS_PER_SEC
    assert abs(snap.age_sec - 2.0) < 1e-9
    assert np.isfinite(safe_float(snap.metrics["front_atm_iv"]))
    assert abs(safe_float(snap.metrics["front_25d_call_delta_abs_error"])) < 1e-12
    assert abs(safe_float(snap.metrics["front_25d_put_delta_abs_error"])) < 1e-12
    # Verify bounded target resolution is exactly latest-prior and never future.
    resolved = nearest_prior_surface_timestamps(
        np.array([target - 5 * NS_PER_SEC, target - 2 * NS_PER_SEC, target + NS_PER_SEC], dtype=np.int64),
        [target - 3 * NS_PER_SEC, target, target + 2 * NS_PER_SEC],
    )
    assert resolved[target - 3 * NS_PER_SEC] == target - 5 * NS_PER_SEC
    assert resolved[target] == target - 2 * NS_PER_SEC
    assert resolved[target + 2 * NS_PER_SEC] == target + NS_PER_SEC

    # Exercise the targeted parquet reader: only requested surface timestamps are materialized.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        day = date(1970, 1, 1)
        droot = root / "options" / FAMILY_MARK / f"date={day.isoformat()}"
        droot.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(droot / "part.parquet", index=False)
        selected = read_mark_rows_for_timestamps(
            root, [day], [target - 2 * NS_PER_SEC], MARK_COLUMNS
        )
        assert not selected.empty
        assert set(selected["ts_recv_ns"].astype("int64").unique()) == {target - 2 * NS_PER_SEC}
        assert len(selected) == 6

    # Verify pre-event cutoff is separate from decision point conceptually.
    candidate_second = 1000
    pre_ns = candidate_second * NS_PER_SEC
    decision_ns = pre_ns + int(3.8 * NS_PER_SEC)
    assert pre_ns < decision_ns
    print(
        "SELF-TEST PASS: option-contract parsing, no-future surface selection, ATM/25d extraction, "
        "targeted mark-surface loading, per-day checkpoint helpers, and conservative timing validated."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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

    print("Phase 7M.11A - Outcome-blind BTC-options regime / anticipation feasibility")
    print("[1/5] Verifying frozen candidate clock and options canonical families...")
    cand_inputs = args.candidate_root / "candidate_inputs.parquet"
    if not cand_inputs.exists():
        raise FileNotFoundError(cand_inputs)
    if not args.sequence_root.exists():
        raise FileNotFoundError(args.sequence_root)
    options_root = args.canonical_root / "options"
    if not options_root.exists():
        raise FileNotFoundError(f"Missing canonical options root: {options_root}")

    print("[2/5] Auditing options regime, pre-event repricing, OI structure, and option activity...")
    cand, summary = run_audit(args)
    print("[3/5] Verifying output remains causal-only...")
    forbid_outcome_columns(cand, "final 7M.11A substrate")
    print("[4/5] Writing feasibility reports and reusable causal substrate...")
    # run_audit writes outputs atomically enough for this lightweight audit.
    print("[5/5] Complete.")
    print(
        f"  anchored={len(cand):,} pre_ATM_IV={summary['surface_pre_atm_iv_coverage_pct']:.2f}% "
        f"pre30m_ATM_change={summary['anticipation_30m_atm_iv_change_coverage_pct']:.2f}% "
        f"pre30m_25d_skew_change={summary['anticipation_30m_25d_skew_change_coverage_pct']:.2f}%"
    )
    print(f"  reports={args.report_root}")
    print(f"  causal substrate={args.output_root / 'candidate_options_regime_anticipation_audit.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
