#!/usr/bin/env python3
"""Phase 7M.8A — Outcome-blind mark/index-basis feasibility audit.

Research question
-----------------
After Phase 7M.2–7M.7 investigated visible-book mechanics, executed-trade
composition, open-interest change, and observed force-order activity, does the
perpetual contract's mark-vs-index dislocation provide a genuinely different
causal information family worth testing?

This phase is FEASIBILITY ONLY. It does not open future outcome labels, does not
match candidates, and does not choose an outcome-facing threshold.

Frozen causal clock
-------------------
For each Phase 7M.6F candidate with a valid 3-second anchor, the decision point
is exactly:

    anchor_ts_recv_ns + 3 seconds

Every markPrice event used here must satisfy ``ts_recv_ns <= target_ts_recv_ns``.
No future interpolation is allowed. For trailing 5/15/30/60-second basis
changes, the earlier endpoint is the latest event available at or before the
corresponding causal target; its staleness is retained explicitly rather than
hidden behind a freshness assumption.

Primary quantities audited (NOT yet promoted as research exposures)
---------------------------------------------------------------------
* mark-index basis in basis points:
      10,000 * (mark_price / index_price - 1)
* direction-aligned current basis:
      candidate_direction * basis_bps
* trailing basis change:
      basis_now - basis_at_or_before(decision - window)
* direction-aligned trailing basis change:
      candidate_direction * trailing_basis_change_bps
* funding-rate state/change, audited only for availability/variation

Interpretation guardrails
-------------------------
* mark-index basis is a perp-vs-reference dislocation measure, not proof of
  causality or participant identity.
* funding rate is a contract-level state variable; this audit does not assume
  it moves on a useful seconds/minutes timescale.
* no future outcomes are opened in this script.
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


SCRIPT_VERSION = "1.0.0"
PHASE = "7M.8A"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
AUDIT_WINDOWS_SEC = (5, 15, 30, 60)
NANOSECONDS_PER_SECOND = 1_000_000_000

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
    "prospective_25bp_mark_index_basis_feasibility_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_mark_index_basis_feasibility"
)

MARK_FAMILY = "mark_price"
MARK_REQUIRED_COLUMNS = (
    "symbol",
    "ts_recv_ns",
    "event_id",
    "source_file_id",
    "source_record_number",
    "exchange_event_ms",
    "mark_price",
    "index_price",
)
MARK_OPTIONAL_COLUMNS = (
    "alternate_mark_price",
    "estimated_settle_price",
    "funding_rate",
    "next_funding_time_ms",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "outcome", "future", "reach_", "p75", "p50", "p100",
    "h300_", "h600_", "h900_", "h1800_", "success",
)


# ---------------------------------------------------------------------------
# General utilities
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


def qstats(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    labels = ("min", "p10", "p25", "median", "p75", "p90", "p99", "max")
    if len(a) == 0:
        return {k: math.nan for k in labels}
    qs = np.quantile(a, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
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
        if any(token in str(c).lower() for token in FORBIDDEN_OUTCOME_TOKENS)
    ]
    if bad:
        raise RuntimeError(f"{context}: future/outcome-like columns are forbidden: {sorted(bad)}")


def read_existing_columns(path: Path, requested: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(requested))
    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    cols = [c for c in requested if c in present]
    if not cols:
        return pd.DataFrame(columns=list(requested))
    out = pd.read_parquet(path, columns=cols)
    for c in requested:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(requested)]


def family_files(canonical_root: Path, family: str, day: date) -> list[Path]:
    directory = canonical_root / "perps" / family / f"date={day.isoformat()}"
    return sorted(directory.glob("*.parquet")) if directory.exists() else []


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


# ---------------------------------------------------------------------------
# Frozen candidate clock
# ---------------------------------------------------------------------------


def load_candidate_inputs(candidate_root: Path) -> pd.DataFrame:
    path = candidate_root / "candidate_inputs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen candidate inputs: {path}")
    frame = pd.read_parquet(path)
    required = {"candidate_id", "candidate_date", "direction", "candidate_second"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen candidate inputs missing {sorted(missing)}")
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("Frozen candidate inputs contain duplicate candidate_id")
    if len(frame) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_CANDIDATES:,} candidates; found {len(frame):,}")
    frame = frame.copy()
    frame["candidate_date"] = frame["candidate_date"].astype(str)
    frame["direction"] = pd.to_numeric(frame["direction"], errors="raise").astype(int)
    frame["candidate_second"] = pd.to_numeric(frame["candidate_second"], errors="raise").astype("int64")
    if not set(frame["direction"].unique()).issubset({-1, 1}):
        raise RuntimeError("Candidate direction must be +/-1")
    forbid_outcome_columns(frame, "candidate_inputs")
    return frame.sort_values(["candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True)


def load_primary_anchor_rows(sequence_root: Path) -> pd.DataFrame:
    paths = sorted(sequence_root.glob("candidate_metrics/candidate_date=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(
            "No Phase 7M.6F candidate_metrics partitions found under "
            f"{sequence_root / 'candidate_metrics'}"
        )
    parts: list[pd.DataFrame] = []
    requested = [
        "candidate_id", "candidate_date", "observation_horizon_sec",
        "anchor_found", "anchor_ts_recv_ns", "horizon_complete", "invalid_reason",
    ]
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
    anchor_ns = pd.to_numeric(out["anchor_ts_recv_ns"], errors="coerce")
    anchor_ok = anchor_found.fillna(False).astype(bool) & np.isfinite(anchor_ns)
    out["anchor_valid_3s"] = anchor_ok
    out["decision_ts_recv_ns"] = np.where(
        anchor_ok,
        anchor_ns + PRIMARY_HORIZON_SEC * NANOSECONDS_PER_SECOND,
        np.nan,
    )
    return out


# ---------------------------------------------------------------------------
# Canonical markPrice loading and causal lookup
# ---------------------------------------------------------------------------


def inspect_mark_schema(canonical_root: Path, study_days: Sequence[date]) -> pd.DataFrame:
    all_fields = list(MARK_REQUIRED_COLUMNS) + list(MARK_OPTIONAL_COLUMNS)
    counts = {field: 0 for field in all_fields}
    files_seen = 0
    total_rows = 0
    schemas: set[tuple[str, ...]] = set()
    for day in study_days:
        for path in family_files(canonical_root, MARK_FAMILY, day):
            pf = pq.ParquetFile(path)
            names = tuple(pf.schema_arrow.names)
            schemas.add(names)
            files_seen += 1
            total_rows += int(pf.metadata.num_rows)
            present = set(names)
            for field in all_fields:
                if field in present:
                    counts[field] += 1
    rows: list[dict[str, Any]] = []
    for field in all_fields:
        rows.append({
            "field": field,
            "required_for_primary_basis": field in {"symbol", "ts_recv_ns", "mark_price", "index_price"},
            "files_present": counts[field],
            "files_total": files_seen,
            "file_presence_frac": (counts[field] / files_seen) if files_seen else math.nan,
        })
    rows.append({
        "field": "__AUDIT_TOTALS__",
        "required_for_primary_basis": False,
        "files_present": files_seen,
        "files_total": files_seen,
        "file_presence_frac": 1.0 if files_seen else math.nan,
        "canonical_rows": total_rows,
        "distinct_schema_shapes": len(schemas),
    })
    return pd.DataFrame(rows)


def _load_mark_day(canonical_root: Path, symbol: str, day: date) -> pd.DataFrame:
    paths = family_files(canonical_root, MARK_FAMILY, day)
    requested = list(MARK_REQUIRED_COLUMNS) + list(MARK_OPTIONAL_COLUMNS)
    parts: list[pd.DataFrame] = []
    for path in paths:
        present = set(pq.ParquetFile(path).schema_arrow.names)
        required_core = {"symbol", "ts_recv_ns", "mark_price", "index_price"}
        missing_core = required_core - present
        if missing_core:
            raise RuntimeError(f"{path}: mark_price canonical file missing {sorted(missing_core)}")
        f = read_existing_columns(path, requested)
        f = f[f["symbol"].astype(str).str.upper().eq(symbol.upper())].copy()
        if not f.empty:
            parts.append(f)
    if not parts:
        return pd.DataFrame(columns=requested)

    out = pd.concat(parts, ignore_index=True)
    out["ts_recv_ns"] = pd.to_numeric(out["ts_recv_ns"], errors="coerce")
    out = out[np.isfinite(out["ts_recv_ns"])].copy()
    out["ts_recv_ns"] = out["ts_recv_ns"].astype("int64")

    # Dedupe reconnect/replay duplicates by the canonical event identity when
    # available, otherwise fall back to the receive/source ordering identity.
    if "event_id" in out.columns and out["event_id"].notna().any():
        sort_cols = [c for c in ("ts_recv_ns", "source_file_id", "source_record_number") if c in out.columns]
        out = out.sort_values(sort_cols, kind="mergesort")
        nonnull = out["event_id"].notna()
        with_id = out[nonnull].drop_duplicates("event_id", keep="last")
        without_id = out[~nonnull]
        out = pd.concat([with_id, without_id], ignore_index=True)
    sort_cols = [c for c in ("ts_recv_ns", "source_file_id", "source_record_number") if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    for col in ("mark_price", "index_price", "alternate_mark_price", "estimated_settle_price", "funding_rate"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "next_funding_time_ms" in out.columns:
        out["next_funding_time_ms"] = pd.to_numeric(out["next_funding_time_ms"], errors="coerce")
    if "exchange_event_ms" in out.columns:
        out["exchange_event_ms"] = pd.to_numeric(out["exchange_event_ms"], errors="coerce")

    good = (
        np.isfinite(out["mark_price"].to_numpy(float)) &
        np.isfinite(out["index_price"].to_numpy(float)) &
        (out["mark_price"].to_numpy(float) > 0) &
        (out["index_price"].to_numpy(float) > 0)
    )
    out["basis_valid"] = good
    out["basis_bps"] = np.where(
        good,
        10_000.0 * (out["mark_price"].to_numpy(float) / out["index_price"].to_numpy(float) - 1.0),
        np.nan,
    )
    return out


@dataclass
class MarkDayCache:
    canonical_root: Path
    symbol: str
    max_days: int = 5

    def __post_init__(self) -> None:
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def day(self, d: date) -> pd.DataFrame:
        key = d.isoformat()
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = _load_mark_day(self.canonical_root, self.symbol, d)
        self._cache[key] = value
        while len(self._cache) > self.max_days:
            self._cache.popitem(last=False)
        return value

    def context(self, center: date) -> pd.DataFrame:
        parts = [self.day(center - timedelta(days=1)), self.day(center), self.day(center + timedelta(days=1))]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).sort_values("ts_recv_ns", kind="mergesort").reset_index(drop=True)


def prior_index(ts: np.ndarray, target_ns: int) -> int:
    """Index of latest ts <= target, or -1 when no prior event exists."""
    return int(np.searchsorted(ts, target_ns, side="right") - 1)


def attach_mark_basis(frame: pd.DataFrame, canonical_root: Path, symbol: str) -> pd.DataFrame:
    valid = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    cache = MarkDayCache(canonical_root, symbol)

    for day_s, group in valid.groupby("candidate_date", sort=True):
        center = date.fromisoformat(str(day_s))
        marks = cache.context(center)
        if marks.empty:
            for r in group.itertuples(index=False):
                rows.append({"candidate_id": r.candidate_id, "mark_event_found": False})
            continue

        ts = marks["ts_recv_ns"].to_numpy(np.int64)
        basis = marks["basis_bps"].to_numpy(float)
        mark_price = marks["mark_price"].to_numpy(float)
        index_price = marks["index_price"].to_numpy(float)
        funding = pd.to_numeric(marks.get("funding_rate", np.nan), errors="coerce").to_numpy(float)
        exch = pd.to_numeric(marks.get("exchange_event_ms", np.nan), errors="coerce").to_numpy(float)
        next_funding = pd.to_numeric(marks.get("next_funding_time_ms", np.nan), errors="coerce").to_numpy(float)

        for r in group.itertuples(index=False):
            decision_ns = int(r.decision_ts_recv_ns)
            idx_now = prior_index(ts, decision_ns)
            base: dict[str, Any] = {"candidate_id": r.candidate_id}
            if idx_now < 0:
                base["mark_event_found"] = False
                rows.append(base)
                continue

            now_ts = int(ts[idx_now])
            now_basis = float(basis[idx_now]) if math.isfinite(float(basis[idx_now])) else math.nan
            now_funding = float(funding[idx_now]) if math.isfinite(float(funding[idx_now])) else math.nan
            base.update({
                "mark_event_found": True,
                "mark_ts_recv_ns": now_ts,
                "mark_age_sec": (decision_ns - now_ts) / 1e9,
                "mark_exchange_event_ms": float(exch[idx_now]) if math.isfinite(float(exch[idx_now])) else math.nan,
                "mark_price": float(mark_price[idx_now]) if math.isfinite(float(mark_price[idx_now])) else math.nan,
                "index_price": float(index_price[idx_now]) if math.isfinite(float(index_price[idx_now])) else math.nan,
                "basis_valid": math.isfinite(now_basis),
                "basis_bps": now_basis,
                "directional_basis_bps": float(r.direction) * now_basis if math.isfinite(now_basis) else math.nan,
                "funding_rate": now_funding,
                "directional_funding_rate": float(r.direction) * now_funding if math.isfinite(now_funding) else math.nan,
                "next_funding_time_ms": float(next_funding[idx_now]) if math.isfinite(float(next_funding[idx_now])) else math.nan,
            })

            for window in AUDIT_WINDOWS_SEC:
                target_ns = decision_ns - window * NANOSECONDS_PER_SECOND
                idx_prev = prior_index(ts, target_ns)
                prefix = f"w{window}s"
                if idx_prev < 0:
                    base[f"{prefix}_prior_found"] = False
                    continue
                prev_ts = int(ts[idx_prev])
                prev_basis = float(basis[idx_prev]) if math.isfinite(float(basis[idx_prev])) else math.nan
                prev_funding = float(funding[idx_prev]) if math.isfinite(float(funding[idx_prev])) else math.nan
                base[f"{prefix}_prior_found"] = True
                base[f"{prefix}_prior_ts_recv_ns"] = prev_ts
                base[f"{prefix}_prior_target_lag_sec"] = (target_ns - prev_ts) / 1e9
                base[f"{prefix}_actual_span_sec"] = (now_ts - prev_ts) / 1e9
                base[f"{prefix}_prior_basis_bps"] = prev_basis
                base[f"{prefix}_basis_change_bps"] = (
                    now_basis - prev_basis if math.isfinite(now_basis) and math.isfinite(prev_basis) else math.nan
                )
                chg = base[f"{prefix}_basis_change_bps"]
                base[f"{prefix}_directional_basis_change_bps"] = (
                    float(r.direction) * chg if math.isfinite(float(chg)) else math.nan
                )
                base[f"{prefix}_prior_funding_rate"] = prev_funding
                base[f"{prefix}_funding_rate_change"] = (
                    now_funding - prev_funding
                    if math.isfinite(now_funding) and math.isfinite(prev_funding) else math.nan
                )
            rows.append(base)

    features = pd.DataFrame(rows)
    out = frame.merge(features, on="candidate_id", how="left", validate="one_to_one")
    if "mark_event_found" not in out.columns:
        out["mark_event_found"] = False
    out["mark_event_found"] = out["mark_event_found"].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Feasibility summaries
# ---------------------------------------------------------------------------


def summarize_daily_cadence(canonical_root: Path, symbol: str, study_days: Sequence[date]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in study_days:
        marks = _load_mark_day(canonical_root, symbol, day)
        if marks.empty:
            rows.append({"date": day.isoformat(), "rows": 0, "valid_basis_rows": 0})
            continue
        ts = marks["ts_recv_ns"].to_numpy(np.int64)
        diffs = np.diff(ts).astype(float) / 1e9 if len(ts) > 1 else np.array([], dtype=float)
        diffs = diffs[diffs >= 0]
        q = qstats(diffs)
        funding = pd.to_numeric(marks.get("funding_rate", np.nan), errors="coerce").to_numpy(float)
        funding_valid = funding[np.isfinite(funding)]
        funding_changes = int(np.sum(np.diff(funding_valid) != 0)) if len(funding_valid) > 1 else 0
        rows.append({
            "date": day.isoformat(),
            "rows": len(marks),
            "valid_basis_rows": int(pd.to_numeric(marks["basis_bps"], errors="coerce").notna().sum()),
            "median_gap_sec": q["median"],
            "p90_gap_sec": q["p90"],
            "p99_gap_sec": q["p99"],
            "max_gap_sec": q["max"],
            "funding_valid_rows": int(len(funding_valid)),
            "funding_distinct_values": int(len(np.unique(funding_valid))) if len(funding_valid) else 0,
            "funding_adjacent_changes": funding_changes,
        })
    return pd.DataFrame(rows)


def summarize_candidate_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["anchor_valid_3s"]].copy()
    rows: list[dict[str, Any]] = []
    found = valid["mark_event_found"].fillna(False).astype(bool)
    rows.append({
        "metric": "decision_mark_event",
        "eligible_candidates": len(valid),
        "available_candidates": int(found.sum()),
        "coverage_frac": float(found.mean()) if len(valid) else math.nan,
        **{f"age_{k}_sec": v for k, v in qstats(pd.to_numeric(valid.loc[found, "mark_age_sec"], errors="coerce")).items()},
    })
    basis_valid = found & pd.to_numeric(valid.get("basis_bps", np.nan), errors="coerce").notna()
    rows.append({
        "metric": "decision_valid_mark_index_basis",
        "eligible_candidates": len(valid),
        "available_candidates": int(basis_valid.sum()),
        "coverage_frac": float(basis_valid.mean()) if len(valid) else math.nan,
        **{f"age_{k}_sec": v for k, v in qstats(pd.to_numeric(valid.loc[basis_valid, "mark_age_sec"], errors="coerce")).items()},
    })
    for window in AUDIT_WINDOWS_SEC:
        col = f"w{window}s_prior_found"
        prior = valid.get(col, False)
        if not isinstance(prior, pd.Series):
            prior = pd.Series(False, index=valid.index)
        prior = prior.fillna(False).astype(bool) & found
        lag_col = f"w{window}s_prior_target_lag_sec"
        span_col = f"w{window}s_actual_span_sec"
        row = {
            "metric": f"basis_change_{window}s_endpoint",
            "eligible_candidates": len(valid),
            "available_candidates": int(prior.sum()),
            "coverage_frac": float(prior.mean()) if len(valid) else math.nan,
        }
        row.update({f"prior_target_lag_{k}_sec": v for k, v in qstats(pd.to_numeric(valid.loc[prior, lag_col], errors="coerce")).items()})
        row.update({f"actual_span_{k}_sec": v for k, v in qstats(pd.to_numeric(valid.loc[prior, span_col], errors="coerce")).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_basis_variation(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["anchor_valid_3s"] & frame["mark_event_found"]].copy()
    metrics: list[tuple[str, str]] = [
        ("basis_bps", "current_mark_minus_index_basis_bps"),
        ("directional_basis_bps", "direction_aligned_current_basis_bps"),
    ]
    for window in AUDIT_WINDOWS_SEC:
        metrics.extend([
            (f"w{window}s_basis_change_bps", f"basis_change_{window}s_bps"),
            (f"w{window}s_directional_basis_change_bps", f"direction_aligned_basis_change_{window}s_bps"),
        ])
    rows: list[dict[str, Any]] = []
    for col, label in metrics:
        if col not in valid.columns:
            continue
        x = pd.to_numeric(valid[col], errors="coerce")
        finite = np.isfinite(x.to_numpy(float))
        values = x.to_numpy(float)[finite]
        row = {
            "metric": label,
            "available_candidates": int(finite.sum()),
            "coverage_frac": float(finite.mean()) if len(valid) else math.nan,
            "positive_frac": float(np.mean(values > 0)) if len(values) else math.nan,
            "negative_frac": float(np.mean(values < 0)) if len(values) else math.nan,
            "zero_frac": float(np.mean(values == 0)) if len(values) else math.nan,
            "nonzero_frac": float(np.mean(values != 0)) if len(values) else math.nan,
        }
        row.update(qstats(values))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_funding_variation(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["anchor_valid_3s"] & frame["mark_event_found"]].copy()
    metrics: list[tuple[str, str]] = [
        ("funding_rate", "current_funding_rate"),
        ("directional_funding_rate", "direction_aligned_current_funding_rate"),
    ]
    for window in AUDIT_WINDOWS_SEC:
        metrics.append((f"w{window}s_funding_rate_change", f"funding_rate_change_{window}s"))
    rows: list[dict[str, Any]] = []
    for col, label in metrics:
        if col not in valid.columns:
            continue
        x = pd.to_numeric(valid[col], errors="coerce")
        values = x.to_numpy(float)
        values = values[np.isfinite(values)]
        row = {
            "metric": label,
            "available_candidates": len(values),
            "coverage_frac": len(values) / len(valid) if len(valid) else math.nan,
            "distinct_values": int(len(np.unique(values))) if len(values) else 0,
            "positive_frac": float(np.mean(values > 0)) if len(values) else math.nan,
            "negative_frac": float(np.mean(values < 0)) if len(values) else math.nan,
            "zero_frac": float(np.mean(values == 0)) if len(values) else math.nan,
            "nonzero_frac": float(np.mean(values != 0)) if len(values) else math.nan,
        }
        row.update(qstats(values))
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_representation(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["anchor_valid_3s"] & frame["mark_event_found"]].copy()
    rows: list[dict[str, Any]] = []
    for split_name, mask in (
        ("ALL", np.ones(len(valid), dtype=bool)),
        ("DISCOVERY", valid["candidate_date"].astype(str) <= "2026-06-18"),
        ("VALIDATION", valid["candidate_date"].astype(str) > "2026-06-18"),
        ("UP", pd.to_numeric(valid["direction"], errors="coerce") == 1),
        ("DOWN", pd.to_numeric(valid["direction"], errors="coerce") == -1),
    ):
        subset = valid.loc[np.asarray(mask)]
        row = {"subgroup": split_name, "candidates": len(subset)}
        for window in AUDIT_WINDOWS_SEC:
            col = f"w{window}s_directional_basis_change_bps"
            x = pd.to_numeric(subset.get(col, np.nan), errors="coerce")
            vals = np.asarray(x, dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"dir_basis_change_{window}s_available"] = len(vals)
            row[f"dir_basis_change_{window}s_positive_frac"] = float(np.mean(vals > 0)) if len(vals) else math.nan
            row[f"dir_basis_change_{window}s_negative_frac"] = float(np.mean(vals < 0)) if len(vals) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_findings(
    frame: pd.DataFrame,
    schema: pd.DataFrame,
    cadence: pd.DataFrame,
    coverage: pd.DataFrame,
    basis: pd.DataFrame,
    funding: pd.DataFrame,
) -> str:
    valid = int(frame["anchor_valid_3s"].sum())
    found = int((frame["anchor_valid_3s"] & frame["mark_event_found"]).sum())
    lines = [
        f"# Phase {PHASE} — Mark/index-basis feasibility",
        "",
        "## Outcome-blind status",
        "",
        "- Future outcomes opened: **NO**",
        "- Matching performed: **NO**",
        "- Composite signal built: **NO**",
        f"- Frozen candidates: **{len(frame):,}**",
        f"- Valid anchor+3s candidates: **{valid:,}**",
        f"- Candidates with causal markPrice state: **{found:,}**",
        "",
        "## What this audit measures",
        "",
        "The audit treats `mark_price` and `index_price` as contemporaneous inputs from the persisted perps `markPrice` stream and computes `10,000 * (mark/index - 1)` in basis points. Trailing changes use only events available at or before each causal endpoint. No future interpolation is used.",
        "",
        "## Next decision",
        "",
        "Use `mark_price_field_presence.csv`, `mark_price_cadence_by_day.csv`, `candidate_endpoint_coverage.csv`, `basis_variation.csv`, and `funding_variation.csv` to decide whether current basis, direction-aligned trailing basis change, and/or funding state have enough coverage and variation to promote into a separate matching-feasibility phase. Do not choose thresholds from outcomes.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    ts = np.asarray([100, 200, 300, 500], dtype=np.int64)
    assert prior_index(ts, 99) == -1
    assert prior_index(ts, 100) == 0
    assert prior_index(ts, 250) == 1
    assert prior_index(ts, 500) == 3
    assert prior_index(ts, 999) == 3

    mark = 101.0
    index = 100.0
    basis = 10_000.0 * (mark / index - 1.0)
    assert abs(basis - 100.0) < 1e-9
    assert abs((-1) * basis + 100.0) < 1e-9

    candidates = pd.DataFrame({
        "candidate_id": ["x"], "candidate_date": ["2026-05-18"],
        "direction": [1], "candidate_second": [100],
    })
    anchors = pd.DataFrame({
        "candidate_id": ["x"], "anchor_found": [True],
        "anchor_ts_recv_ns": [100_900_000_000], "horizon_complete": [True],
        "invalid_reason": [None],
    })
    clock = build_causal_candidate_clock(candidates, anchors)
    assert int(clock.loc[0, "decision_ts_recv_ns"]) == 103_900_000_000

    print("SELF-TEST PASS: exact anchor+3s clock, latest-prior causal lookup, basis formula, and direction alignment validated.")


# ---------------------------------------------------------------------------
# CLI / main
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

    print(f"Phase {PHASE} - Outcome-blind perps mark/index-basis feasibility")
    print("[1/6] Loading frozen 7M.3 candidate universe and exact 7M.6F anchor+3s clock...")
    candidates = load_candidate_inputs(args.candidate_root)
    anchors = load_primary_anchor_rows(args.sequence_root)
    frame = build_causal_candidate_clock(candidates, anchors)
    valid = frame[frame["anchor_valid_3s"]]
    if valid.empty:
        raise RuntimeError("No valid anchor+3s candidates")
    study_start = date.fromisoformat(str(valid["candidate_date"].min()))
    study_end = date.fromisoformat(str(valid["candidate_date"].max()))
    study_days = [study_start + timedelta(days=i) for i in range((study_end - study_start).days + 1)]
    print(f"      candidates={len(frame):,} anchor+3s={len(valid):,} dates={study_start}..{study_end}")

    print("[2/6] Verifying persisted canonical markPrice fields and daily cadence...")
    schema = inspect_mark_schema(args.canonical_root, study_days)
    total_files = int(schema.loc[schema["field"].eq("__AUDIT_TOTALS__"), "files_total"].iloc[0]) if (schema["field"] == "__AUDIT_TOTALS__").any() else 0
    if total_files == 0:
        raise FileNotFoundError(
            f"No canonical perps/{MARK_FAMILY}/date=YYYY-MM-DD parquet files found under {args.canonical_root}"
        )
    required_presence = schema[schema["required_for_primary_basis"].eq(True)]
    bad_required = required_presence[required_presence["files_present"] < required_presence["files_total"]]
    if not bad_required.empty:
        raise RuntimeError(
            "Primary mark/index fields are not present in every discovered canonical mark_price file: "
            + ", ".join(bad_required["field"].astype(str))
        )
    cadence = summarize_daily_cadence(args.canonical_root, args.symbol, study_days)

    print("[3/6] Attaching exact-causal mark/index/funding state to candidate decision clocks...")
    frame = attach_mark_basis(frame, args.canonical_root, args.symbol)
    found = int((frame["anchor_valid_3s"] & frame["mark_event_found"]).sum())
    print(f"      causal markPrice found={found:,}/{int(frame['anchor_valid_3s'].sum()):,}")

    print("[4/6] Computing outcome-blind coverage, staleness, basis variation, and funding variation...")
    coverage = summarize_candidate_coverage(frame)
    basis = summarize_basis_variation(frame)
    funding = summarize_funding_variation(frame)
    subgroups = subgroup_representation(frame)

    print("[5/6] Writing reusable causal audit substrate and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    forbid_outcome_columns(frame, "7M.8A candidate mark/index audit output")
    causal_path = args.output_root / "candidate_mark_index_basis_audit.parquet"
    frame.to_parquet(causal_path, index=False)

    write_csv(args.report_root / "mark_price_field_presence.csv", schema)
    write_csv(args.report_root / "mark_price_cadence_by_day.csv", cadence)
    write_csv(args.report_root / "candidate_endpoint_coverage.csv", coverage)
    write_csv(args.report_root / "basis_variation.csv", basis)
    write_csv(args.report_root / "funding_variation.csv", funding)
    write_csv(args.report_root / "subgroup_representation.csv", subgroups)

    invariants = pd.DataFrame([
        {"invariant": "future_outcomes_opened", "value": False},
        {"invariant": "matching_performed", "value": False},
        {"invariant": "composite_signal_built", "value": False},
        {"invariant": "decision_clock", "value": "exact_7m6f_anchor_ts_recv_ns_plus_3s"},
        {"invariant": "mark_event_causality", "value": "latest_ts_recv_ns<=target_ts_recv_ns"},
        {"invariant": "future_interpolation", "value": False},
        {"invariant": "freshness_threshold_imposed", "value": False},
        {"invariant": "basis_definition", "value": "10000*(mark_price/index_price-1)"},
        {"invariant": "directional_basis_definition", "value": "candidate_direction*basis_bps"},
        {"invariant": "candidate_count", "value": len(frame)},
        {"invariant": "anchor_valid_3s_count", "value": int(frame["anchor_valid_3s"].sum())},
        {"invariant": "causal_mark_event_count", "value": found},
    ])
    write_csv(args.report_root / "analysis_invariants.csv", invariants)

    findings = make_findings(frame, schema, cadence, coverage, basis, funding)
    (args.report_root / "findings.md").write_text(findings, encoding="utf-8")

    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "symbol": args.symbol,
        "candidate_count": len(frame),
        "anchor_valid_3s_count": int(frame["anchor_valid_3s"].sum()),
        "causal_mark_event_count": found,
        "study_start": study_start.isoformat(),
        "study_end": study_end.isoformat(),
        "audit_windows_sec": list(AUDIT_WINDOWS_SEC),
        "future_outcomes_opened": False,
        "matching_performed": False,
        "composite_signal_built": False,
        "freshness_threshold_imposed": False,
        "candidate_input_sha256": sha256_file(args.candidate_root / "candidate_inputs.parquet"),
        "causal_output": str(causal_path),
        "reports": {
            "field_presence": str(args.report_root / "mark_price_field_presence.csv"),
            "cadence": str(args.report_root / "mark_price_cadence_by_day.csv"),
            "candidate_coverage": str(args.report_root / "candidate_endpoint_coverage.csv"),
            "basis_variation": str(args.report_root / "basis_variation.csv"),
            "funding_variation": str(args.report_root / "funding_variation.csv"),
            "subgroup_representation": str(args.report_root / "subgroup_representation.csv"),
            "findings": str(args.report_root / "findings.md"),
        },
    }
    write_json(args.report_root / "summary.json", summary)

    print("[6/6] COMPLETE - future outcomes remain unopened.")
    print(f"      causal substrate: {causal_path}")
    print(f"      reports:          {args.report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
