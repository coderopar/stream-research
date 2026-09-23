#!/usr/bin/env python3
"""Phase 7M.12B - outcome-blind options gamma matching feasibility.

Research question
-----------------
After imposing a strict causal freshness/completeness gate, can prospective
25-bp BTC-perp candidates that begin in HIGH versus LOW observed near-spot
unsigned gamma-sensitivity environments be matched cleanly on the already
frozen early-event perp mechanics?

Primary exposure
----------------
Fresh observed near-spot unsigned gamma sensitivity:

    sum(open_interest * abs(gamma)) for option strikes within +/-2% of spot

The exposure is recomputed in this phase using only recent OI observations.
The 7M.12A 30-minute OI window is NOT reused as the scientific exposure
window; it remains an audit ceiling only.

Frozen causal data-quality gate
-------------------------------
* mark/Greek surface is the exact latest surface at or before the START of
  candidate_second and must be <= 2.0 seconds old;
* each OI observation used in the exposure must be <= 90 seconds old;
* fresh +/-2% OI must retain >= 95% of the broader 7M.12A near-spot OI mass,
  preventing stale contracts from silently disappearing from the exposure;
* >= 95% of that fresh near-spot OI mass must overlap contracts with a finite
  contemporaneous gamma value.

These thresholds are fixed from feed cadence / 7M.12A coverage diagnostics,
not from continuation outcomes. No future outcome file is opened here.

Exposure states
---------------
Within each exact candidate_date x direction stratum, among candidates passing
the gamma-quality gate:
* LOW  = exposure <= the stratum 1/3 quantile;
* HIGH = exposure >= the stratum 2/3 quantile;
* middle third is excluded;
* if the two quantile boundaries are not distinct, that stratum is not used.

This relative definition avoids treating calendar-level gamma drift as the
scientific treatment and does not use outcome information.

Matching
--------
Exact candidate_date + direction, one-to-one without replacement, using the
same five frozen early-perp controls used in 7M.11B:
* 3s directional mid-price move;
* 3s directional aggressive quote;
* opposing peak initial queue-loss share;
* opposing observed refill quote;
* opposing refill breadth.

Matching discipline is unchanged from 7M.11B:
* per-control |z difference| <= 0.75;
* Euclidean z distance <= 1.00;
* repair until max |SMD| <= 0.20;
* full pairs >= 250;
* discovery pairs >= 100;
* validation pairs >= 100.

Interpretation boundary
-----------------------
A PASS means the unsigned observed gamma context has enough common support for
a later outcome phase. It does NOT mean gamma predicts continuation, does NOT
identify actual hedge flow, and does NOT infer dealer/net gamma sign from
public OI.
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
PHASE = "7M.12B"
EXPECTED_7M12A_VERSION = "1.0.0"
EXPECTED_7M11A_HELPER_VERSION = "1.0.1"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"
NS_PER_SEC = 1_000_000_000
EPS = 1e-12

# Outcome-blind data-quality thresholds frozen for this phase.
MARK_MAX_AGE_SEC = 2.0
OI_MAX_AGE_SEC = 90
NEAR_SPOT_BAND_PCT = 2.0
MIN_FRESH_OI_RETENTION_SHARE = 0.95
MIN_GAMMA_OI_COVERAGE_SHARE = 0.95

# Matching discipline inherited unchanged from 7M.11B.
MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20
MIN_FULL_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_REPAIR_PAIRS = 8

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M12A_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_options_gamma_landscape_feasibility_v1"
)
DEFAULT_7M12A_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_options_gamma_landscape_feasibility"
)
DEFAULT_7M6F_ROOT = (
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
    "prospective_25bp_options_gamma_matching_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_25bp_options_gamma_matching_feasibility"
)

AUDIT_COLUMNS = [
    "candidate_id", "candidate_date", "period", "direction", "candidate_second",
    "pre_event_cutoff_ns", "anchor_valid_3s", "mark_surface_age_sec",
    "oi_recent_near_2pct_sum", "gamma_oi_abs_near_2pct",
    "gamma_oi_abs_delta_change_1pct_near_2pct",
]
MECHANICS_COLUMNS = [
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "anchor_found", "sequence_valid", "horizon_complete",
    "opposing__observed_refill_quote", "opposing__refilled_level_count",
    "control__directional_mid_move_bps_3s", "control__directional_aggressive_quote_3s",
    "control__opposing_peak_initial_queue_loss_share_3s", "is_reaccel_context",
]
BASE_CONTROL_COLUMNS = ["__momentum", "__aggression", "__destruction", "__refill", "__refill_breadth"]
FORBIDDEN_TOKENS = (
    "outcome", "future", "p75", "p50", "p100", "mfe", "mae",
    "first_passage", "reach_", "reaches_", "h300_", "h600_", "h900_",
    "h1800_", "continuation_target", "success",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.12B requires pandas, numpy and pyarrow in the project .venv.")


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
    if not (math.isfinite(num) and math.isfinite(den)) or den <= 0:
        return math.nan
    return float(num / den)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0.0
    return s.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def split_period(day_text: str) -> str:
    return "DISCOVERY" if str(day_text) <= DISCOVERY_END_DATE else "VALIDATION"


def assert_causal_names(columns: Iterable[str], context: str) -> None:
    bad = []
    for c in columns:
        lc = str(c).lower()
        if any(tok in lc for tok in FORBIDDEN_TOKENS):
            bad.append(str(c))
    if bad:
        raise RuntimeError(f"{context} exposes forbidden future/outcome-like columns: {sorted(bad)}")


def qstats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    keys = ("min", "p10", "p25", "median", "p75", "p90", "p99", "max")
    if len(arr) == 0:
        return {k: math.nan for k in keys}
    vals = np.quantile(arr, [0, .10, .25, .50, .75, .90, .99, 1])
    return dict(zip(keys, map(float, vals)))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Source-contract verification / loading
# ---------------------------------------------------------------------------


def verify_7m12a_contract(audit_root: Path, report_root: Path) -> dict[str, Any]:
    summary_path = report_root / "summary.json"
    audit_path = audit_root / "candidate_options_gamma_landscape_audit.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing 7M.12A summary: {summary_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing 7M.12A candidate audit: {audit_path}")
    meta = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(meta.get("phase")) != "7M.12A":
        raise RuntimeError(f"Expected 7M.12A summary; found phase={meta.get('phase')}")
    if str(meta.get("script_version")) != EXPECTED_7M12A_VERSION:
        raise RuntimeError(
            f"Expected 7M.12A v{EXPECTED_7M12A_VERSION}; found v{meta.get('script_version')}"
        )
    if int(meta.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        raise RuntimeError("7M.12A candidate-count contract mismatch")
    if bool(meta.get("future_outcomes_opened", True)):
        raise RuntimeError("7M.12A summary does not certify outcome-blind construction")
    expected_sha = str(meta.get("candidate_audit_sha256", ""))
    actual_sha = sha256_file(audit_path)
    if not expected_sha or expected_sha != actual_sha:
        raise RuntimeError("7M.12A candidate-audit SHA256 does not match its frozen summary")
    return {"summary": meta, "audit_path": audit_path, "audit_sha256": actual_sha}


def read_required_columns(path: Path, requested: Sequence[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    missing = [c for c in requested if c not in names]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    out = pq.read_table(path, columns=list(requested)).to_pandas()
    assert_causal_names(out.columns, str(path))
    return out


def load_7m12a(audit_path: Path) -> pd.DataFrame:
    d = read_required_columns(audit_path, AUDIT_COLUMNS)
    d["candidate_id"] = d["candidate_id"].astype(str)
    d["candidate_date"] = d["candidate_date"].astype(str)
    d["direction"] = pd.to_numeric(d["direction"], errors="raise").astype(int)
    d["pre_event_cutoff_ns"] = pd.to_numeric(d["pre_event_cutoff_ns"], errors="raise").astype("int64")
    if len(d) != EXPECTED_CANDIDATES or d["candidate_id"].duplicated().any():
        raise RuntimeError("7M.12A audit is not the expected unique 7,067-candidate universe")
    return d


def load_7m6f_primary(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("candidate_metrics/candidate_date=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No 7M.6F candidate_metrics under {root}")
    parts: list[pd.DataFrame] = []
    for p in paths:
        pf = pq.ParquetFile(p)
        names = set(pf.schema_arrow.names)
        missing = [c for c in MECHANICS_COLUMNS if c not in names]
        if missing:
            raise RuntimeError(f"{p} missing 7M.6F causal columns: {missing}")
        f = pq.read_table(p, columns=MECHANICS_COLUMNS).to_pandas()
        h = pd.to_numeric(f["observation_horizon_sec"], errors="coerce")
        f = f[h == PRIMARY_HORIZON_SEC].copy()
        if not f.empty:
            parts.append(f)
    if not parts:
        raise RuntimeError("No primary 3s 7M.6F mechanics rows found")
    out = pd.concat(parts, ignore_index=True)
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["candidate_date"] = out["candidate_date"].astype(str)
    if out["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate primary 7M.6F candidate rows")
    assert_causal_names(out.columns, "7M.6F mechanics")
    return out


def load_7m11a_helper(path: Path):
    mod = load_module(path, "phase7m11a_helper_for_7m12b")
    if str(getattr(mod, "SCRIPT_VERSION", "")) != EXPECTED_7M11A_HELPER_VERSION:
        raise RuntimeError(
            f"Expected 7M.11A helper v{EXPECTED_7M11A_HELPER_VERSION}; "
            f"found v{getattr(mod, 'SCRIPT_VERSION', '?')}"
        )
    required = [
        "MarkTimestampCache", "nearest_prior_surface_timestamps",
        "read_mark_rows_for_timestamps", "numericize_mark", "DayCache", "numericize_oi",
    ]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"7M.11A helper missing required functions/classes: {missing}")
    return mod


# ---------------------------------------------------------------------------
# Strict fresh gamma reconstruction
# ---------------------------------------------------------------------------


class CausalOiCursor:
    """Latest OI per contract using only rows at-or-before monotone causal targets."""

    def __init__(self, oi: pd.DataFrame):
        self.frame = oi.sort_values(["ts_recv_ns", "symbol"], kind="mergesort").reset_index(drop=True).copy()
        self.times = self.frame["ts_recv_ns"].to_numpy(dtype=np.int64) if not self.frame.empty else np.array([], dtype=np.int64)
        self.symbols = self.frame["symbol"].astype(str).to_numpy() if not self.frame.empty else np.array([], dtype=object)
        self.pos = 0
        self.latest_idx: dict[str, int] = {}
        self.last_target = -1

    def latest(self, target_ns: int, max_age_sec: int) -> pd.DataFrame:
        target_ns = int(target_ns)
        if target_ns < self.last_target:
            raise RuntimeError("CausalOiCursor targets must be monotonically nondecreasing")
        self.last_target = target_ns
        while self.pos < len(self.times) and int(self.times[self.pos]) <= target_ns:
            self.latest_idx[str(self.symbols[self.pos])] = self.pos
            self.pos += 1
        lo = target_ns - int(max_age_sec * NS_PER_SEC)
        idxs = [i for i in self.latest_idx.values() if int(self.times[i]) > lo]
        if not idxs:
            return self.frame.iloc[0:0].copy()
        return self.frame.iloc[idxs].copy()


def strict_gamma_metrics(
    mark_surface: pd.DataFrame,
    fresh_oi: pd.DataFrame,
    target_ns: int,
    surface_ts_ns: int | None,
    broad_near2_oi_sum: float,
) -> dict[str, Any]:
    out = {
        "fresh_mark_surface_found": False,
        "fresh_mark_surface_ts_recv_ns": surface_ts_ns,
        "fresh_mark_surface_age_sec": math.nan,
        "fresh_spot_index_price": math.nan,
        "fresh_oi_contracts": 0,
        "fresh_near2_oi_contracts": 0,
        "fresh_near2_oi_sum": math.nan,
        "fresh_near2_oi_retention_share": math.nan,
        "fresh_near2_oi_age_median_sec": math.nan,
        "fresh_near2_oi_age_p90_sec": math.nan,
        "fresh_near2_oi_age_max_sec": math.nan,
        "fresh_near2_gamma_contracts": 0,
        "fresh_near2_gamma_oi_coverage_share": math.nan,
        "gamma_oi_abs_near_2pct_fresh": math.nan,
        "gamma_oi_abs_delta_change_1pct_near_2pct_fresh": math.nan,
        "gamma_quality_pass": False,
        "gamma_quality_fail_reason": "NO_MARK_SURFACE",
    }
    if surface_ts_ns is None or mark_surface.empty:
        return out

    age = float((int(target_ns) - int(surface_ts_ns)) / NS_PER_SEC)
    out["fresh_mark_surface_found"] = True
    out["fresh_mark_surface_age_sec"] = age
    if not math.isfinite(age) or age < 0 or age > MARK_MAX_AGE_SEC:
        out["gamma_quality_fail_reason"] = "MARK_SURFACE_STALE"
        return out

    mark = mark_surface.copy()
    assert_causal_names(mark.columns, "canonical options mark/Greeks")
    idx = pd.to_numeric(mark.get("index_price"), errors="coerce")
    idx = idx[np.isfinite(idx) & (idx > 0)]
    if idx.empty:
        out["gamma_quality_fail_reason"] = "NO_VALID_SPOT_INDEX"
        return out
    spot = float(idx.median())
    out["fresh_spot_index_price"] = spot

    oi = fresh_oi.copy()
    assert_causal_names(oi.columns, "canonical options OI")
    if oi.empty:
        out["gamma_quality_fail_reason"] = "NO_FRESH_OI"
        return out
    oi["open_interest"] = pd.to_numeric(oi.get("open_interest"), errors="coerce")
    oi["strike"] = pd.to_numeric(oi.get("strike"), errors="coerce")
    oi = oi[np.isfinite(oi["open_interest"]) & (oi["open_interest"] >= 0)].copy()
    out["fresh_oi_contracts"] = int(oi["symbol"].nunique())
    if oi.empty:
        out["gamma_quality_fail_reason"] = "NO_VALID_FRESH_OI"
        return out

    near = oi[
        np.isfinite(oi["strike"])
        & (np.abs(oi["strike"] / spot - 1.0) * 100.0 <= NEAR_SPOT_BAND_PCT + 1e-12)
    ].copy()
    if near.empty:
        out["gamma_quality_fail_reason"] = "NO_FRESH_NEAR2_OI"
        return out
    near_sum = float(near["open_interest"].sum())
    out["fresh_near2_oi_contracts"] = int(near["symbol"].nunique())
    out["fresh_near2_oi_sum"] = near_sum
    ages = (int(target_ns) - near["ts_recv_ns"].to_numpy(dtype=np.int64)) / NS_PER_SEC
    out["fresh_near2_oi_age_median_sec"] = float(np.median(ages)) if len(ages) else math.nan
    out["fresh_near2_oi_age_p90_sec"] = float(np.quantile(ages, .90)) if len(ages) else math.nan
    out["fresh_near2_oi_age_max_sec"] = float(np.max(ages)) if len(ages) else math.nan

    broad = safe_float(broad_near2_oi_sum)
    retention = safe_div(near_sum, broad)
    out["fresh_near2_oi_retention_share"] = retention
    if not math.isfinite(retention) or retention + EPS < MIN_FRESH_OI_RETENTION_SHARE:
        out["gamma_quality_fail_reason"] = "STALE_OI_MASS_GAP"
        return out

    mark = mark.sort_values(["symbol", "ts_recv_ns"], kind="mergesort").groupby(
        "symbol", as_index=False, sort=False
    ).tail(1).copy()
    mark["gamma"] = pd.to_numeric(mark.get("gamma"), errors="coerce")
    joined = near.merge(
        mark[["symbol", "gamma"]], on="symbol", how="left", validate="one_to_one"
    )
    gamma_ok = np.isfinite(joined["gamma"])
    gamma_mass = float(joined.loc[gamma_ok, "open_interest"].sum())
    coverage = safe_div(gamma_mass, near_sum)
    out["fresh_near2_gamma_oi_coverage_share"] = coverage
    out["fresh_near2_gamma_contracts"] = int(joined.loc[gamma_ok, "symbol"].nunique())
    if not math.isfinite(coverage) or coverage + EPS < MIN_GAMMA_OI_COVERAGE_SHARE:
        out["gamma_quality_fail_reason"] = "INSUFFICIENT_GAMMA_OI_OVERLAP"
        return out

    good = joined[gamma_ok].copy()
    good["gamma_oi_abs"] = good["open_interest"] * np.abs(good["gamma"])
    good = good[np.isfinite(good["gamma_oi_abs"]) & (good["gamma_oi_abs"] >= 0)].copy()
    if good.empty:
        out["gamma_quality_fail_reason"] = "NO_VALID_GAMMA_EXPOSURE"
        return out
    exposure = float(good["gamma_oi_abs"].sum())
    if not math.isfinite(exposure) or exposure <= 0:
        out["gamma_quality_fail_reason"] = "NONPOSITIVE_GAMMA_EXPOSURE"
        return out

    out["gamma_oi_abs_near_2pct_fresh"] = exposure
    out["gamma_oi_abs_delta_change_1pct_near_2pct_fresh"] = exposure * spot * 0.01
    out["gamma_quality_pass"] = True
    out["gamma_quality_fail_reason"] = "PASS"
    return out


def checkpoint_paths(output_root: Path, day_text: str) -> tuple[Path, Path]:
    root = output_root / "fresh_gamma_day_checkpoints" / f"candidate_date={day_text}"
    return root / "part.parquet", root / "manifest.json"


def load_checkpoint(
    output_root: Path,
    day_text: str,
    source_audit_sha: str,
) -> pd.DataFrame | None:
    part, manifest = checkpoint_paths(output_root, day_text)
    if not (part.exists() and manifest.exists()):
        return None
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    checks = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "contract_version": CHECKPOINT_CONTRACT_VERSION,
        "candidate_date": day_text,
        "source_7m12a_audit_sha256": source_audit_sha,
        "mark_max_age_sec": MARK_MAX_AGE_SEC,
        "oi_max_age_sec": OI_MAX_AGE_SEC,
        "near_spot_band_pct": NEAR_SPOT_BAND_PCT,
        "min_fresh_oi_retention_share": MIN_FRESH_OI_RETENTION_SHARE,
        "min_gamma_oi_coverage_share": MIN_GAMMA_OI_COVERAGE_SHARE,
    }
    for k, v in checks.items():
        if meta.get(k) != v:
            return None
    if str(meta.get("part_sha256", "")) != sha256_file(part):
        return None
    frame = pd.read_parquet(part)
    assert_causal_names(frame.columns, "7M.12B checkpoint")
    if len(frame) != int(meta.get("candidate_rows", -1)):
        return None
    return frame


def save_checkpoint(
    output_root: Path,
    day_text: str,
    frame: pd.DataFrame,
    source_audit_sha: str,
) -> None:
    part, manifest = checkpoint_paths(output_root, day_text)
    atomic_parquet(part, frame)
    atomic_json(manifest, {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "contract_version": CHECKPOINT_CONTRACT_VERSION,
        "candidate_date": day_text,
        "candidate_rows": int(len(frame)),
        "source_7m12a_audit_sha256": source_audit_sha,
        "mark_max_age_sec": MARK_MAX_AGE_SEC,
        "oi_max_age_sec": OI_MAX_AGE_SEC,
        "near_spot_band_pct": NEAR_SPOT_BAND_PCT,
        "min_fresh_oi_retention_share": MIN_FRESH_OI_RETENTION_SHARE,
        "min_gamma_oi_coverage_share": MIN_GAMMA_OI_COVERAGE_SHARE,
        "created_at_utc": utc_now_iso(),
        "part_sha256": sha256_file(part),
    })


def rebuild_fresh_gamma(
    audit: pd.DataFrame,
    canonical_root: Path,
    helper_script: Path,
    output_root: Path,
    source_audit_sha: str,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    helper = load_7m11a_helper(helper_script)
    mark_ts_cache = helper.MarkTimestampCache(canonical_root, max_days=3)
    oi_cache = helper.DayCache(canonical_root, max_days=2)
    parts: list[pd.DataFrame] = []
    day_rows: list[dict[str, Any]] = []

    days = sorted(audit["candidate_date"].unique())
    for i, day_text in enumerate(days, start=1):
        g = audit[audit["candidate_date"] == day_text].copy().sort_values(
            ["pre_event_cutoff_ns", "candidate_id"], kind="mergesort"
        )
        cached = None if force else load_checkpoint(output_root, day_text, source_audit_sha)
        if cached is not None:
            parts.append(cached)
            passed = bool_series(cached["gamma_quality_pass"])
            day_rows.append({
                "candidate_date": day_text,
                "candidates": int(len(cached)),
                "gamma_quality_pass": int(passed.sum()),
                "checkpoint_reused": True,
            })
            print(f"  [{i:02d}/{len(days):02d}] {day_text}: quality={int(passed.sum()):4d}/{len(cached):4d} checkpoint=REUSED")
            continue

        day = date.fromisoformat(day_text)
        valid_targets = g.loc[bool_series(g["anchor_valid_3s"]), "pre_event_cutoff_ns"].astype("int64").tolist()
        all_ts, _cur_ts, canonical_mark_rows = mark_ts_cache.load_window(day)
        resolved = helper.nearest_prior_surface_timestamps(all_ts, valid_targets)
        needed_ts = sorted({int(v) for v in resolved.values() if v is not None})
        marks = helper.read_mark_rows_for_timestamps(
            canonical_root,
            [day - timedelta(days=1), day],
            needed_ts,
            helper.MARK_COLUMNS,
        )
        marks = helper.numericize_mark(marks) if not marks.empty else marks
        mark_groups = {int(ts): x.copy() for ts, x in marks.groupby("ts_recv_ns", sort=False)} if not marks.empty else {}

        oi = oi_cache.load_window("open_interest", day, helper.OI_COLUMNS, include_prev=True)
        oi = helper.numericize_oi(oi)
        oi_cursor = CausalOiCursor(oi)

        rows: list[dict[str, Any]] = []
        for r in g.itertuples(index=False):
            base = {
                "candidate_id": str(r.candidate_id),
                "candidate_date": str(r.candidate_date),
                "direction": int(r.direction),
                "pre_event_cutoff_ns": int(r.pre_event_cutoff_ns),
                "anchor_valid_3s": bool(r.anchor_valid_3s),
                "broad_near2_oi_sum_7m12a": safe_float(r.oi_recent_near_2pct_sum),
                "audit_gamma_oi_abs_near_2pct": safe_float(r.gamma_oi_abs_near_2pct),
            }
            if not base["anchor_valid_3s"]:
                base.update(strict_gamma_metrics(pd.DataFrame(), pd.DataFrame(), base["pre_event_cutoff_ns"], None, math.nan))
                base["gamma_quality_fail_reason"] = "INVALID_3S_ANCHOR"
                rows.append(base)
                continue
            target = int(base["pre_event_cutoff_ns"])
            surface_ts = resolved.get(target)
            surface = mark_groups.get(int(surface_ts), pd.DataFrame()) if surface_ts is not None else pd.DataFrame()
            fresh_oi = oi_cursor.latest(target, OI_MAX_AGE_SEC)
            metrics = strict_gamma_metrics(
                surface, fresh_oi, target, surface_ts, base["broad_near2_oi_sum_7m12a"]
            )
            base.update(metrics)
            rows.append(base)

        out = pd.DataFrame(rows)
        assert_causal_names(out.columns, "7M.12B fresh gamma checkpoint")
        save_checkpoint(output_root, day_text, out, source_audit_sha)
        parts.append(out)
        passed = bool_series(out["gamma_quality_pass"])
        day_rows.append({
            "candidate_date": day_text,
            "candidates": int(len(out)),
            "gamma_quality_pass": int(passed.sum()),
            "checkpoint_reused": False,
            "canonical_mark_rows": int(canonical_mark_rows),
            "needed_mark_surfaces": int(len(needed_ts)),
            "loaded_mark_rows": int(len(marks)),
        })
        print(f"  [{i:02d}/{len(days):02d}] {day_text}: quality={int(passed.sum()):4d}/{len(out):4d} checkpoint=SAVED")

    fresh = pd.concat(parts, ignore_index=True)
    if len(fresh) != EXPECTED_CANDIDATES or fresh["candidate_id"].duplicated().any():
        raise RuntimeError("Strict fresh-gamma reconstruction did not preserve the frozen candidate universe")
    return fresh, pd.DataFrame(day_rows)


# ---------------------------------------------------------------------------
# Mechanics / exposure state freeze
# ---------------------------------------------------------------------------


def build_analysis_frame(audit: pd.DataFrame, fresh: pd.DataFrame, mechanics: pd.DataFrame) -> pd.DataFrame:
    # Mechanics must cover the full frozen candidate universe; invalid anchors remain explicit.
    check = audit[["candidate_id", "candidate_date", "direction"]].merge(
        mechanics[["candidate_id", "candidate_date", "direction"]],
        on="candidate_id", how="inner", validate="one_to_one", suffixes=("_a", "_m")
    )
    if len(check) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"7M.6F primary mechanics cover {len(check):,}/{EXPECTED_CANDIDATES:,} frozen candidates"
        )
    if not (check["candidate_date_a"].astype(str) == check["candidate_date_m"].astype(str)).all():
        raise RuntimeError("Candidate-date mismatch between 7M.12A and 7M.6F")
    if not (
        pd.to_numeric(check["direction_a"], errors="coerce")
        == pd.to_numeric(check["direction_m"], errors="coerce")
    ).all():
        raise RuntimeError("Direction mismatch between 7M.12A and 7M.6F")

    fresh_keep = [c for c in fresh.columns if c not in {"candidate_date", "direction", "pre_event_cutoff_ns", "anchor_valid_3s"}]
    d = audit.merge(fresh[fresh_keep], on="candidate_id", how="left", validate="one_to_one")
    mech_keep = [c for c in MECHANICS_COLUMNS if c not in {"candidate_id", "candidate_date", "direction", "observation_horizon_sec"}]
    d = d.merge(mechanics[["candidate_id", *mech_keep]], on="candidate_id", how="left", validate="one_to_one")

    d["base_mechanics_eligible"] = (
        bool_series(d["anchor_found"])
        & bool_series(d["sequence_valid"])
        & bool_series(d["horizon_complete"])
    )
    d["__momentum"] = pd.to_numeric(d["control__directional_mid_move_bps_3s"], errors="coerce")
    d["__aggression"] = np.log1p(pd.to_numeric(d["control__directional_aggressive_quote_3s"], errors="coerce").clip(lower=0))
    d["__destruction"] = pd.to_numeric(d["control__opposing_peak_initial_queue_loss_share_3s"], errors="coerce")
    d["__refill"] = np.log1p(pd.to_numeric(d["opposing__observed_refill_quote"], errors="coerce").clip(lower=0))
    d["__refill_breadth"] = np.log1p(pd.to_numeric(d["opposing__refilled_level_count"], errors="coerce").clip(lower=0))
    assert_causal_names(d.columns, "7M.12B analysis frame")
    return d


def freeze_relative_tertiles(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = d.copy()
    exposure = pd.to_numeric(x["gamma_oi_abs_near_2pct_fresh"], errors="coerce")
    quality = bool_series(x["gamma_quality_pass"]) & np.isfinite(exposure) & (exposure > 0)
    x["gamma_exposure_state"] = pd.Series(pd.NA, index=x.index, dtype="object")
    x["exp_high_vs_low_gamma_near2"] = np.nan
    x["gamma_stratum_q33"] = np.nan
    x["gamma_stratum_q67"] = np.nan

    rows: list[dict[str, Any]] = []
    for (day, direction), idx in x[quality].groupby(["candidate_date", "direction"], sort=True).groups.items():
        vals = exposure.loc[idx].astype(float)
        q33 = float(vals.quantile(1.0 / 3.0, interpolation="linear"))
        q67 = float(vals.quantile(2.0 / 3.0, interpolation="linear"))
        distinct = bool(math.isfinite(q33) and math.isfinite(q67) and q67 > q33 + EPS)
        x.loc[idx, "gamma_stratum_q33"] = q33
        x.loc[idx, "gamma_stratum_q67"] = q67
        low_idx: list[int] = []
        high_idx: list[int] = []
        mid_idx: list[int] = []
        if distinct:
            low_idx = vals[vals <= q33 + EPS].index.tolist()
            high_idx = vals[vals >= q67 - EPS].index.tolist()
            overlap = set(low_idx).intersection(high_idx)
            if overlap:
                raise RuntimeError("High/low gamma tertiles overlap inside a supposedly distinct stratum")
            middle = set(idx) - set(low_idx) - set(high_idx)
            mid_idx = sorted(middle)
            x.loc[low_idx, "gamma_exposure_state"] = "LOW"
            x.loc[low_idx, "exp_high_vs_low_gamma_near2"] = 0.0
            x.loc[high_idx, "gamma_exposure_state"] = "HIGH"
            x.loc[high_idx, "exp_high_vs_low_gamma_near2"] = 1.0
            x.loc[mid_idx, "gamma_exposure_state"] = "MIDDLE_EXCLUDED"
        else:
            x.loc[list(idx), "gamma_exposure_state"] = "NONDISTINCT_STRATUM_EXCLUDED"
        rows.append({
            "candidate_date": str(day),
            "direction": int(direction),
            "quality_candidates": int(len(idx)),
            "q33": q33,
            "q67": q67,
            "boundaries_distinct": distinct,
            "low_candidates": int(len(low_idx)),
            "middle_candidates": int(len(mid_idx)),
            "high_candidates": int(len(high_idx)),
        })
    return x, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Matching (discipline inherited from 7M.11B)
# ---------------------------------------------------------------------------


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if x.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    sd = float(x.std(ddof=1))
    mu = float(x.mean())
    if not math.isfinite(sd) or sd <= EPS:
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    den = math.sqrt(max(0.0, 0.5 * (float(np.var(x, ddof=1)) + float(np.var(y, ddof=1)))))
    if den <= EPS:
        return 0.0 if math.isclose(float(x.mean()), float(y.mean()), abs_tol=EPS) else math.inf
    return float((x.mean() - y.mean()) / den)


def matching_eligible(d: pd.DataFrame) -> pd.DataFrame:
    z = d[bool_series(d["gamma_quality_pass"]) & bool_series(d["base_mechanics_eligible"])].copy()
    z = z[pd.to_numeric(z["exp_high_vs_low_gamma_near2"], errors="coerce").notna()].copy()
    z = z[np.isfinite(pd.to_numeric(z["gamma_oi_abs_near_2pct_fresh"], errors="coerce"))].copy()
    for c in BASE_CONTROL_COLUMNS:
        z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()
    return z


def direct_binary_pairs(d: pd.DataFrame) -> pd.DataFrame:
    x = matching_eligible(d)
    exp = pd.to_numeric(x["exp_high_vs_low_gamma_near2"], errors="coerce")
    vals = set(exp.dropna().astype(int).unique().tolist())
    if not vals.issubset({0, 1}):
        raise RuntimeError(f"Non-binary gamma exposure values: {sorted(vals)}")
    for c in BASE_CONTROL_COLUMNS:
        x[f"__z__{c}"] = zscore(x[c])

    high = x[exp == 1].copy()
    low = x[exp == 0].copy()
    rows: list[dict[str, Any]] = []
    for (dt, direction), hb in high.groupby(["candidate_date", "direction"], sort=True):
        lb = low[
            (low["candidate_date"].astype(str) == str(dt))
            & (pd.to_numeric(low["direction"], errors="coerce") == int(direction))
        ].copy()
        available = set(lb.index)
        for _, hr in hb.sort_values(["candidate_id"], kind="mergesort").iterrows():
            best = None
            for li in sorted(available):
                lr = lb.loc[li]
                diffs: list[float] = []
                okay = True
                for c in BASE_CONTROL_COLUMNS:
                    hv = safe_float(hr[f"__z__{c}"])
                    lv = safe_float(lr[f"__z__{c}"])
                    if not math.isfinite(hv) or not math.isfinite(lv) or abs(hv - lv) > MATCH_Z_CALIPER:
                        okay = False
                        break
                    diffs.append(hv - lv)
                if not okay:
                    continue
                dist = math.sqrt(sum(v * v for v in diffs))
                if dist > MATCH_EUCLIDEAN_CALIPER:
                    continue
                key = (dist, str(lr["candidate_id"]))
                if best is None or key < best[0]:
                    best = (key, li, lr, diffs)
            if best is None:
                continue
            _, li, lr, diffs = best
            available.remove(li)
            row: dict[str, Any] = {
                "candidate_date": str(dt),
                "period": split_period(str(dt)),
                "direction": int(direction),
                "high_candidate_id": str(hr["candidate_id"]),
                "low_candidate_id": str(lr["candidate_id"]),
                "high_gamma_oi_abs_near_2pct_fresh": safe_float(hr["gamma_oi_abs_near_2pct_fresh"]),
                "low_gamma_oi_abs_near_2pct_fresh": safe_float(lr["gamma_oi_abs_near_2pct_fresh"]),
                "high_gamma_delta_change_1pct_proxy": safe_float(hr["gamma_oi_abs_delta_change_1pct_near_2pct_fresh"]),
                "low_gamma_delta_change_1pct_proxy": safe_float(lr["gamma_oi_abs_delta_change_1pct_near_2pct_fresh"]),
                "distance": float(best[0][0]),
            }
            for c, diff in zip(BASE_CONTROL_COLUMNS, diffs):
                row[f"high__{c}"] = safe_float(hr[c])
                row[f"low__{c}"] = safe_float(lr[c])
                row[f"diff__{c}"] = float(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def pair_balance(pairs: pd.DataFrame) -> tuple[float, dict[str, float]]:
    vals: dict[str, float] = {}
    for c in BASE_CONTROL_COLUMNS:
        vals[c] = smd(
            pairs.get(f"high__{c}", pd.Series(dtype=float)),
            pairs.get(f"low__{c}", pd.Series(dtype=float)),
        )
    finite = [abs(v) for v in vals.values() if math.isfinite(v)]
    return (max(finite) if finite else math.nan), vals


def repair_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    x = pairs.copy().reset_index(drop=True)
    while len(x) > MIN_REPAIR_PAIRS:
        mx, vals = pair_balance(x)
        if math.isfinite(mx) and mx <= BALANCE_SMD_GATE:
            break
        finite = {k: v for k, v in vals.items() if math.isfinite(v)}
        if not finite:
            break
        worst = max(finite, key=lambda c: (abs(finite[c]), c))
        sign = 1 if finite[worst] >= 0 else -1
        contribution = (
            pd.to_numeric(x[f"high__{worst}"], errors="coerce")
            - pd.to_numeric(x[f"low__{worst}"], errors="coerce")
        ) * sign
        drop_idx = contribution.fillna(-np.inf).idxmax()
        x = x.drop(index=drop_idx).reset_index(drop=True)
    return x


def evaluate_matching(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = matching_eligible(d)
    exp = pd.to_numeric(eligible["exp_high_vs_low_gamma_near2"], errors="coerce").astype(int)
    pre = direct_binary_pairs(d)
    post = repair_pairs(pre) if len(pre) else pre
    mx, smds = pair_balance(post) if len(post) else (math.nan, {})
    disc = int((post["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(post) else 0
    val = int((post["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(post) else 0
    feasible = bool(
        len(post) >= MIN_FULL_PAIRS
        and disc >= MIN_SPLIT_PAIRS
        and val >= MIN_SPLIT_PAIRS
        and math.isfinite(mx)
        and mx <= BALANCE_SMD_GATE
    )
    gate = pd.DataFrame([{
        "family": "OPTIONS_NEAR2_UNSIGNED_GAMMA_FRESH",
        "priority": "PRIMARY",
        "exposure_column": "exp_high_vs_low_gamma_near2",
        "eligible_candidates": int(len(eligible)),
        "high_candidates": int((exp == 1).sum()),
        "low_candidates": int((exp == 0).sum()),
        "pairs_before_repair": int(len(pre)),
        "pairs_after_repair": int(len(post)),
        "discovery_pairs": disc,
        "validation_pairs": val,
        "max_abs_smd": mx,
        "full_pair_gate": MIN_FULL_PAIRS,
        "split_pair_gate": MIN_SPLIT_PAIRS,
        "balance_gate": BALANCE_SMD_GATE,
        "feasible_before_outcomes": feasible,
    }])
    balance = pd.DataFrame([
        {"control": c, "smd": smds.get(c, math.nan), "abs_smd": abs(smds.get(c, math.nan)) if math.isfinite(smds.get(c, math.nan)) else math.nan}
        for c in BASE_CONTROL_COLUMNS
    ])
    return gate, post, balance


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def quality_gate_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(d)
    steps = [
        ("frozen_candidates", pd.Series(True, index=d.index)),
        ("anchor_valid_3s", bool_series(d["anchor_valid_3s"])),
        ("mark_surface_age_le_2s", pd.to_numeric(d["fresh_mark_surface_age_sec"], errors="coerce").between(0, MARK_MAX_AGE_SEC, inclusive="both")),
        ("fresh_near2_oi_retention_ge_95pct", pd.to_numeric(d["fresh_near2_oi_retention_share"], errors="coerce") >= MIN_FRESH_OI_RETENTION_SHARE),
        ("fresh_near2_gamma_oi_coverage_ge_95pct", pd.to_numeric(d["fresh_near2_gamma_oi_coverage_share"], errors="coerce") >= MIN_GAMMA_OI_COVERAGE_SHARE),
        ("gamma_quality_pass", bool_series(d["gamma_quality_pass"])),
        ("base_mechanics_eligible", bool_series(d["base_mechanics_eligible"])),
    ]
    running = pd.Series(True, index=d.index)
    for name, mask in steps:
        running = running & mask.fillna(False)
        rows.append({
            "step": name,
            "standalone_passing": int(mask.fillna(False).sum()),
            "sequential_passing": int(running.sum()),
            "sequential_pct_of_frozen": float(running.mean() * 100.0) if total else math.nan,
        })
    return pd.DataFrame(rows)


def failure_reason_summary(d: pd.DataFrame) -> pd.DataFrame:
    x = d["gamma_quality_fail_reason"].fillna("MISSING").astype(str)
    out = x.value_counts(dropna=False).rename_axis("reason").reset_index(name="candidates")
    out["pct_of_frozen"] = out["candidates"] / len(d) * 100.0 if len(d) else math.nan
    return out


def freshness_distribution(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "fresh_mark_surface_age_sec",
        "fresh_near2_oi_age_median_sec",
        "fresh_near2_oi_age_p90_sec",
        "fresh_near2_oi_age_max_sec",
        "fresh_near2_oi_retention_share",
        "fresh_near2_gamma_oi_coverage_share",
        "gamma_oi_abs_near_2pct_fresh",
        "gamma_oi_abs_delta_change_1pct_near_2pct_fresh",
    ]
    groups = [("FULL", d), ("DISCOVERY", d[d["candidate_date"] <= DISCOVERY_END_DATE]), ("VALIDATION", d[d["candidate_date"] >= VALIDATION_START_DATE])]
    for group, g in groups:
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce")
            finite = vals[np.isfinite(vals)]
            row = {"group": group, "metric": metric, "rows": int(len(g)), "valid": int(len(finite)), "valid_frac": float(len(finite) / len(g)) if len(g) else math.nan}
            row.update(qstats(finite))
            rows.append(row)
    return pd.DataFrame(rows)


def subgroup_representation(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in [
        ("FULL", pd.Series(True, index=d.index)),
        ("DISCOVERY", d["candidate_date"] <= DISCOVERY_END_DATE),
        ("VALIDATION", d["candidate_date"] >= VALIDATION_START_DATE),
        ("UP", pd.to_numeric(d["direction"], errors="coerce") == 1),
        ("DOWN", pd.to_numeric(d["direction"], errors="coerce") == -1),
    ]:
        g = d[mask].copy()
        state = g["gamma_exposure_state"].astype("object")
        rows.append({
            "subgroup": name,
            "candidates": int(len(g)),
            "quality_pass": int(bool_series(g["gamma_quality_pass"]).sum()),
            "high_candidates": int((state == "HIGH").sum()),
            "low_candidates": int((state == "LOW").sum()),
            "middle_excluded": int((state == "MIDDLE_EXCLUDED").sum()),
            "matching_eligible": int(len(matching_eligible(g))),
        })
    return pd.DataFrame(rows)


def make_findings(d: pd.DataFrame, gate: pd.DataFrame) -> str:
    qpass = int(bool_series(d["gamma_quality_pass"]).sum())
    r = gate.iloc[0]
    status = "PASS" if bool(r["feasible_before_outcomes"]) else "FAIL"
    return f"""# Phase 7M.12B — outcome-blind options gamma matching feasibility

## Question

After removing stale/incomplete option states, can HIGH versus LOW **observed unsigned near-spot gamma sensitivity** be matched cleanly on the frozen early-perp mechanics?

## Freshness/completeness gate

- Mark/Greek surface must be <= **{MARK_MAX_AGE_SEC:.1f}s** old.
- Only OI observations <= **{OI_MAX_AGE_SEC}s** old are used in the exposure.
- Fresh +/-{NEAR_SPOT_BAND_PCT:.0f}% OI must retain >= **{MIN_FRESH_OI_RETENTION_SHARE*100:.0f}%** of the broader 7M.12A near-spot OI mass.
- >= **{MIN_GAMMA_OI_COVERAGE_SHARE*100:.0f}%** of fresh near-spot OI must have finite contemporaneous gamma.
- Gamma-quality pass: **{qpass:,}/{len(d):,} ({qpass/len(d)*100:.2f}%)**.

This does not repair stale observations by interpolation or future fill. Stale material is excluded, and candidates with material stale near-spot OI mass fail the gate.

## Exposure

Primary primitive: `sum(OI * abs(gamma))` within +/-2% of spot, recomputed using only fresh OI.

HIGH and LOW are the upper and lower within-`candidate_date x direction` tertiles. The middle third is excluded. No continuation outcome is used to set the boundaries.

## Matching gate

- Eligible high/low candidates: **{int(r['eligible_candidates']):,}**
- Pairs before repair: **{int(r['pairs_before_repair']):,}**
- Pairs after repair: **{int(r['pairs_after_repair']):,}**
- Discovery pairs: **{int(r['discovery_pairs']):,}**
- Validation pairs: **{int(r['validation_pairs']):,}**
- Max |SMD|: **{safe_float(r['max_abs_smd']):.4f}**
- Feasibility gate: **{status}**

## Interpretation

A PASS means the unsigned observed gamma context has adequate common support for a later outcome phase. It does **not** mean high gamma improves or harms continuation, and public OI still does not identify dealer/net gamma sign or actual hedge direction.
"""


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> int:
    target = 1_800_000_000_000_000_000
    surface_ts = target - 800_000_000
    mark = pd.DataFrame([
        {"ts_recv_ns": surface_ts, "symbol": "BTC-260930-99000-C", "index_price": 100000.0, "strike": 99000.0, "gamma": 0.001},
        {"ts_recv_ns": surface_ts, "symbol": "BTC-260930-101000-C", "index_price": 100000.0, "strike": 101000.0, "gamma": 0.002},
    ])
    fresh = pd.DataFrame([
        {"ts_recv_ns": target - 30 * NS_PER_SEC, "symbol": "BTC-260930-99000-C", "strike": 99000.0, "open_interest": 100.0},
        {"ts_recv_ns": target - 60 * NS_PER_SEC, "symbol": "BTC-260930-101000-C", "strike": 101000.0, "open_interest": 50.0},
    ])
    good = strict_gamma_metrics(mark, fresh, target, surface_ts, 150.0)
    assert bool(good["gamma_quality_pass"])
    assert math.isclose(good["gamma_oi_abs_near_2pct_fresh"], 0.2, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(good["fresh_near2_oi_retention_share"], 1.0, rel_tol=0, abs_tol=1e-12)

    # If one-third of the broad near-spot OI is stale/missing, the candidate fails.
    bad_retention = strict_gamma_metrics(mark, fresh.iloc[[0]].copy(), target, surface_ts, 150.0)
    assert not bool(bad_retention["gamma_quality_pass"])
    assert bad_retention["gamma_quality_fail_reason"] == "STALE_OI_MASS_GAP"

    # A stale mark surface fails even if OI is otherwise fresh.
    stale_mark = strict_gamma_metrics(mark, fresh, target, target - 3 * NS_PER_SEC, 150.0)
    assert not bool(stale_mark["gamma_quality_pass"])
    assert stale_mark["gamma_quality_fail_reason"] == "MARK_SURFACE_STALE"

    # Relative tertiles are date x direction local and leave the middle out.
    toy = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(6)],
        "candidate_date": ["2026-06-01"] * 6,
        "direction": [1] * 6,
        "gamma_quality_pass": [True] * 6,
        "gamma_oi_abs_near_2pct_fresh": [1., 2., 3., 4., 5., 6.],
    })
    frozen, strata = freeze_relative_tertiles(toy)
    assert int((frozen["gamma_exposure_state"] == "LOW").sum()) == 2
    assert int((frozen["gamma_exposure_state"] == "HIGH").sum()) == 2
    assert int((frozen["gamma_exposure_state"] == "MIDDLE_EXCLUDED").sum()) == 2
    assert bool(strata.iloc[0]["boundaries_distinct"])

    # Matching remains exact on date/direction and respects calipers.
    mrows = []
    for i in range(12):
        hi = i >= 6
        mrows.append({
            "candidate_id": f"m{i}", "candidate_date": "2026-06-01", "direction": 1,
            "gamma_quality_pass": True, "base_mechanics_eligible": True,
            "exp_high_vs_low_gamma_near2": float(hi),
            "gamma_oi_abs_near_2pct_fresh": float(10 + i),
            "gamma_oi_abs_delta_change_1pct_near_2pct_fresh": float(100 + i),
            "__momentum": float(i % 6), "__aggression": float(i % 6),
            "__destruction": float(i % 6), "__refill": float(i % 6),
            "__refill_breadth": float(i % 6),
        })
    pairs = direct_binary_pairs(pd.DataFrame(mrows))
    assert len(pairs) == 6
    assert (pairs["candidate_date"] == "2026-06-01").all()
    assert (pairs["direction"] == 1).all()

    print(
        "SELF-TEST PASS: strict no-future mark/OI freshness, stale-OI mass retention, "
        "near-spot gamma coverage, within-date-direction tertiles, and inherited matching calipers validated."
    )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root", type=Path, default=DEFAULT_7M12A_ROOT)
    p.add_argument("--audit-report-root", type=Path, default=DEFAULT_7M12A_REPORT_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_7M6F_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--phase7m11a-script", type=Path, default=DEFAULT_7M11A_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--force", action="store_true", help="Ignore reusable per-day fresh-gamma checkpoints")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_stack()

    print(f"{PHASE} - Outcome-blind fresh options-gamma matching feasibility")
    print("[1/6] Verifying frozen 7M.12A audit contract and loading causal mechanics...")
    contract = verify_7m12a_contract(args.audit_root, args.audit_report_root)
    audit = load_7m12a(contract["audit_path"])
    mechanics = load_7m6f_primary(args.sequence_root)

    print("[2/6] Rebuilding +/-2% gamma with strict causal freshness/completeness gates...")
    fresh, day_diag = rebuild_fresh_gamma(
        audit=audit,
        canonical_root=args.canonical_root,
        helper_script=args.phase7m11a_script,
        output_root=args.output_root,
        source_audit_sha=contract["audit_sha256"],
        force=args.force,
    )

    print("[3/6] Freezing within-date x direction HIGH/LOW gamma states...")
    analysis = build_analysis_frame(audit, fresh, mechanics)
    analysis, strata = freeze_relative_tertiles(analysis)

    print("[4/6] Matching HIGH versus LOW gamma candidates on frozen early-perp mechanics...")
    gate, pairs, balance = evaluate_matching(analysis)

    print("[5/6] Freezing pair manifest and outcome-blind reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_root / "candidate_options_gamma_matching_frame.parquet"
    strata_path = args.output_root / "gamma_exposure_strata.parquet"
    pairs_path = args.output_root / "gamma_high_low_pairs.parquet"
    atomic_parquet(analysis_path, analysis)
    atomic_parquet(strata_path, strata)
    atomic_parquet(pairs_path, pairs)

    args.report_root.mkdir(parents=True, exist_ok=True)
    qgate = quality_gate_summary(analysis)
    failures = failure_reason_summary(analysis)
    freshdist = freshness_distribution(analysis)
    subgroup = subgroup_representation(analysis)
    atomic_csv(args.report_root / "quality_gate_summary.csv", qgate)
    atomic_csv(args.report_root / "quality_failure_reasons.csv", failures)
    atomic_csv(args.report_root / "freshness_distribution.csv", freshdist)
    atomic_csv(args.report_root / "gamma_exposure_strata.csv", strata)
    atomic_csv(args.report_root / "subgroup_representation.csv", subgroup)
    atomic_csv(args.report_root / "matching_gate.csv", gate)
    atomic_csv(args.report_root / "matching_balance.csv", balance)
    atomic_csv(args.report_root / "day_reconstruction_diagnostics.csv", day_diag)

    invariants = pd.DataFrame([
        {"invariant": "phase", "value": PHASE},
        {"invariant": "script_version", "value": SCRIPT_VERSION},
        {"invariant": "future_outcomes_opened", "value": False},
        {"invariant": "dealer_or_net_gamma_sign_inferred", "value": False},
        {"invariant": "primary_exposure", "value": "fresh sum(OI*abs(gamma)) within +/-2% of spot"},
        {"invariant": "mark_max_age_sec", "value": MARK_MAX_AGE_SEC},
        {"invariant": "oi_max_age_sec", "value": OI_MAX_AGE_SEC},
        {"invariant": "min_fresh_oi_retention_share", "value": MIN_FRESH_OI_RETENTION_SHARE},
        {"invariant": "min_gamma_oi_coverage_share", "value": MIN_GAMMA_OI_COVERAGE_SHARE},
        {"invariant": "exposure_state_definition", "value": "within exact candidate_date x direction lower/upper tertiles"},
        {"invariant": "middle_third_excluded", "value": True},
        {"invariant": "match_exact", "value": "candidate_date + direction"},
        {"invariant": "match_z_caliper", "value": MATCH_Z_CALIPER},
        {"invariant": "match_euclidean_caliper", "value": MATCH_EUCLIDEAN_CALIPER},
        {"invariant": "balance_smd_gate", "value": BALANCE_SMD_GATE},
        {"invariant": "min_full_pairs", "value": MIN_FULL_PAIRS},
        {"invariant": "min_split_pairs", "value": MIN_SPLIT_PAIRS},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", invariants)
    findings = make_findings(analysis, gate)
    atomic_text(args.report_root / "findings.md", findings)

    r = gate.iloc[0]
    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "scientific_status": "OUTCOME_BLIND_MATCHING_FEASIBILITY_COMPLETE",
        "candidate_count": int(len(analysis)),
        "gamma_quality_pass_count": int(bool_series(analysis["gamma_quality_pass"]).sum()),
        "matching_eligible_count": int(r["eligible_candidates"]),
        "pairs_before_repair": int(r["pairs_before_repair"]),
        "pairs_after_repair": int(r["pairs_after_repair"]),
        "discovery_pairs": int(r["discovery_pairs"]),
        "validation_pairs": int(r["validation_pairs"]),
        "max_abs_smd": safe_float(r["max_abs_smd"]),
        "feasible_before_outcomes": bool(r["feasible_before_outcomes"]),
        "future_outcomes_opened": False,
        "dealer_or_net_gamma_sign_inferred": False,
        "mark_max_age_sec": MARK_MAX_AGE_SEC,
        "oi_max_age_sec": OI_MAX_AGE_SEC,
        "near_spot_band_pct": NEAR_SPOT_BAND_PCT,
        "min_fresh_oi_retention_share": MIN_FRESH_OI_RETENTION_SHARE,
        "min_gamma_oi_coverage_share": MIN_GAMMA_OI_COVERAGE_SHARE,
        "source_7m12a_audit_sha256": contract["audit_sha256"],
        "analysis_frame": str(analysis_path),
        "analysis_frame_sha256": sha256_file(analysis_path),
        "exposure_strata": str(strata_path),
        "exposure_strata_sha256": sha256_file(strata_path),
        "frozen_pair_manifest": str(pairs_path),
        "frozen_pair_manifest_sha256": sha256_file(pairs_path),
    }
    atomic_json(args.report_root / "summary.json", summary)

    print("[6/6] Complete.")
    print(
        f"  gamma quality: {summary['gamma_quality_pass_count']:,}/{summary['candidate_count']:,}; "
        f"pairs={summary['pairs_after_repair']:,} "
        f"D/V={summary['discovery_pairs']:,}/{summary['validation_pairs']:,} "
        f"max|SMD|={summary['max_abs_smd']:.4f} "
        f"PASS={summary['feasible_before_outcomes']}"
    )
    print(f"  frozen pairs: {pairs_path}")
    print(f"  reports:      {args.report_root}")
    print("  Future outcomes remain unopened. Do not proceed to an outcome phase without review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
