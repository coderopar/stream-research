#!/usr/bin/env python3
"""Phase 7M.11B - outcome-blind BTC-options anticipation matching feasibility.

Research question
-----------------
Do three pre-event BTC-options repricing states have enough clean common support
for a later continuation-outcome test, after controlling the already-frozen
prospective 25-bp Binance-perp state?

Frozen families (all measured over the 30 minutes ending BEFORE candidate_second)
-------------------------------------------------------------------------------
PRIMARY - OPTIONS_DIRECTIONAL_25D_SKEW_CHANGE_30M
    high = direction-aligned approximate 25-delta skew change > 0
    low  = direction-aligned approximate 25-delta skew change < 0

    Physical meaning:
      UP: call-vs-put 25d IV spread increased.
      DOWN: call-vs-put 25d IV spread decreased (puts richened vs calls).

    Quality guard (outcome-blind):
      * exact front expiry unchanged over the 30m comparison;
      * at BOTH endpoints, selected call and put deltas must each be within
        0.10 absolute delta of the intended +0.25 / -0.25 targets.
      * exact zero change excluded.

SECONDARY - OPTIONS_ATM_IV_CHANGE_30M
    high = front-expiry ATM IV increased > 0
    low  = front-expiry ATM IV decreased < 0

    Quality guard:
      * exact front expiry unchanged;
      * exact selected front ATM strike unchanged across the 30m comparison;
      * exact zero change excluded.

SECONDARY - OPTIONS_FRONT_END_TERM_RICHENING_30M
    7M.11A stores term as second-expiry ATM IV minus front-expiry ATM IV.
    We orient the exposure as front-end richening:
      front_end_richening = -(change in second_minus_front_atm_iv)
    high = front expiry richened vs second expiry (>0)
    low  = front expiry cheapened vs second expiry (<0)

    Quality guard:
      * exact front expiry unchanged;
      * exact second expiry unchanged;
      * exact zero change excluded.

The older endpoint quality fields were not persisted by 7M.11A, so this script
performs a SMALL TARGETED reread of only the exact pre-event and pre-30m option
surfaces selected by 7M.11A. It does not rescan/materialize whole option days.

Frozen base causal controls
---------------------------
Exact candidate_date + direction, then 1:1 matching on:
* 3s directional perp momentum
* log1p(3s directional perp aggressive quote)
* initial opposing queue destruction
* log1p(anchor-relative 3s opposing refill quote)
* log1p(anchor-relative 3s opposing refill breadth)

Matching constants retained from the 7M lineage:
* per-control |z difference| <= 0.75
* Euclidean z distance <= 1.00
* repair until max |SMD| <= 0.20
* full pairs >= 250
* discovery pairs >= 100
* validation pairs >= 100

Scientific safeguards
---------------------
* 7M.11A v1.0.1 causal-only substrate/report is verified first.
* No future outcome file is opened.
* No options composite signal.
* OI/gamma and option-trade fields are NOT used here; they are reserved for
  later hedging-pressure attribution if an anticipation primitive survives.
* Same-strike call/put IV spread is intentionally excluded (7M.11A found it
  nearly degenerate).
* Event-time option response is intentionally excluded from the anticipation
  families.
* No caliper, quality threshold, or family definition may be relaxed after
  seeing this feasibility report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
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

SCRIPT_VERSION = "1.0.1"
PHASE = "7M.11B"
EXPECTED_7M11A_VERSION = "1.0.1"
EXPECTED_CANDIDATES = 7067
EXPECTED_ANCHORED = 7031
PRIMARY_HORIZON_SEC = 3
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

DELTA_TARGET_MAX_ABS_ERROR = 0.10
MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20
MIN_FULL_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_REPAIR_PAIRS = 8
EPS = 1e-12
NS_PER_SEC = 1_000_000_000

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M11A_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "prospective_25bp_options_regime_anticipation_feasibility_v1"
DEFAULT_7M11A_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_options_regime_anticipation_feasibility"
DEFAULT_7M6F_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "prospective_25bp_refill_redepletion_sequence_v2"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_7M11A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m11a_audit_options_regime_anticipation_feasibility.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "prospective_25bp_options_anticipation_matching_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_options_anticipation_matching_feasibility"

AUDIT_COLUMNS = [
    "candidate_id", "candidate_date", "period", "direction", "candidate_second",
    "pre_event_cutoff_ns", "decision_ts_recv_ns",
    "pre_surface_ts_recv_ns", "pre_minus_30m_surface_ts_recv_ns",
    "anticipation_30m__front_25d_call_put_iv_diff_change",
    "anticipation_30m__front_atm_iv_change",
    "anticipation_30m__second_minus_front_atm_iv_change",
    "anticipation_30m__front_expiry_same",
    "anticipation_30m__front_atm_strike_same",
]
MECHANICS_COLUMNS = [
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "anchor_found", "sequence_valid", "horizon_complete",
    "opposing__observed_refill_quote", "opposing__refilled_level_count",
    "control__directional_mid_move_bps_3s", "control__directional_aggressive_quote_3s",
    "control__opposing_peak_initial_queue_loss_share_3s", "is_reaccel_context",
]
FORBIDDEN_TOKENS = (
    "outcome", "future", "p75", "p50", "p100", "mfe", "mae",
    "first_passage", "reach_", "reaches_", "h300_", "h600_", "h900_",
    "h1800_", "continuation_target", "success",
)
BASE_CONTROL_COLUMNS = ["__momentum", "__aggression", "__destruction", "__refill", "__refill_breadth"]
FAMILY_SPECS = [
    {
        "family": "OPTIONS_DIRECTIONAL_25D_SKEW_CHANGE_30M",
        "priority": "PRIMARY",
        "exposure": "exp_options_directional_skew_change_30m",
        "controls": BASE_CONTROL_COLUMNS,
        "high_state": "DIRECTIONAL_SKEW_ALIGNED",
        "low_state": "DIRECTIONAL_SKEW_COUNTER",
    },
    {
        "family": "OPTIONS_ATM_IV_CHANGE_30M",
        "priority": "SECONDARY",
        "exposure": "exp_options_atm_iv_change_30m",
        "controls": BASE_CONTROL_COLUMNS,
        "high_state": "ATM_IV_EXPANSION",
        "low_state": "ATM_IV_CONTRACTION",
    },
    {
        "family": "OPTIONS_FRONT_END_TERM_RICHENING_30M",
        "priority": "SECONDARY",
        "exposure": "exp_options_front_end_term_richening_30m",
        "controls": BASE_CONTROL_COLUMNS,
        "high_state": "FRONT_END_RICHENING",
        "low_state": "FRONT_END_CHEAPENING",
    },
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.11B requires pandas, numpy and pyarrow in the project .venv.")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False).astype(bool)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_causal_names(columns: Iterable[str], context: str) -> None:
    bad: list[str] = []
    for c in columns:
        cl = str(c).lower()
        if any(tok in cl for tok in FORBIDDEN_TOKENS):
            bad.append(str(c))
    if bad:
        raise RuntimeError(f"{context}: future/outcome-like columns are forbidden: {bad}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m11b-", suffix=path.suffix or ".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m11b-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m11b-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_columns(path: Path, requested: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    missing = [c for c in requested if c not in names]
    if missing:
        raise RuntimeError(f"{path} missing required causal columns: {missing}")
    assert_causal_names(requested, str(path))
    return pq.read_table(path, columns=list(requested)).to_pandas()


def split_period(day_text: str) -> str:
    return "DISCOVERY" if str(day_text) <= DISCOVERY_END_DATE else "VALIDATION"


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# Frozen 7M.11A contract / substrates
# ---------------------------------------------------------------------------

def verify_7m11a_contract(audit_root: Path, report_root: Path) -> dict[str, Any]:
    causal_path = audit_root / "candidate_options_regime_anticipation_audit.parquet"
    summary_path = report_root / "summary.json"
    invariants_path = report_root / "analysis_invariants.csv"
    if not causal_path.exists():
        raise FileNotFoundError(causal_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not invariants_path.exists():
        raise FileNotFoundError(invariants_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "phase": summary.get("phase") == "7M.11A",
        "script_version": str(summary.get("script_version")) == EXPECTED_7M11A_VERSION,
        "candidate_population": int(summary.get("candidate_population", -1)) == EXPECTED_CANDIDATES,
        "anchored_candidates": int(summary.get("anchored_candidates", -1)) == EXPECTED_ANCHORED,
        "future_outcomes_opened": summary.get("future_outcomes_opened") is False,
    }
    inv = pd.read_csv(invariants_path)
    inv_map = {str(r.invariant): str(r.value) for r in inv.itertuples(index=False)}
    checks.update({
        "matching_not_performed": inv_map.get("matching_performed", "").lower() == "false",
        "composite_not_created": inv_map.get("composite_signal_created", "").lower() == "false",
        "pre_event_cutoff": inv_map.get("pre_event_cutoff") == "candidate_second start boundary",
        "decision_point": inv_map.get("decision_point") == "exact Phase 7M.6F anchor + 3s",
        "surface_threshold_not_pre_frozen": inv_map.get("surface_exposure_threshold_frozen", "").lower() == "false",
    })
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise RuntimeError(f"7M.11A contract verification failed: {bad}")
    return {
        "summary": summary,
        "invariants": inv_map,
        "causal_sha256": sha256_file(causal_path),
        "checks": checks,
    }


def load_7m11a(root: Path) -> pd.DataFrame:
    path = root / "candidate_options_regime_anticipation_audit.parquet"
    d = read_columns(path, AUDIT_COLUMNS)
    d["candidate_id"] = d["candidate_id"].astype(str)
    d["candidate_date"] = d["candidate_date"].astype(str)
    d["direction"] = pd.to_numeric(d["direction"], errors="coerce")
    if len(d) != EXPECTED_ANCHORED:
        raise RuntimeError(f"Expected {EXPECTED_ANCHORED} anchored 7M.11A rows, found {len(d)}")
    if d["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate candidate_id in 7M.11A causal substrate")
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


def build_analysis_frame(audit: pd.DataFrame, mechanics: pd.DataFrame) -> pd.DataFrame:
    check = audit[["candidate_id", "candidate_date", "direction"]].merge(
        mechanics[["candidate_id", "candidate_date", "direction"]],
        on="candidate_id", how="inner", validate="one_to_one", suffixes=("_a", "_m")
    )
    if len(check) != EXPECTED_ANCHORED:
        raise RuntimeError("7M.11A anchored candidate universe is not a complete subset of 7M.6F")
    if not (check["candidate_date_a"].astype(str) == check["candidate_date_m"].astype(str)).all():
        raise RuntimeError("Candidate-date mismatch between 7M.11A and 7M.6F")
    if not (
        pd.to_numeric(check["direction_a"], errors="coerce")
        == pd.to_numeric(check["direction_m"], errors="coerce")
    ).all():
        raise RuntimeError("Direction mismatch between 7M.11A and 7M.6F")

    keep = [c for c in MECHANICS_COLUMNS if c not in {"candidate_id", "candidate_date", "direction", "observation_horizon_sec"}]
    projection = ["candidate_id", *keep]
    if len(projection) != len(set(projection)):
        raise RuntimeError(f"Internal duplicate mechanics projection: {projection}")
    d = audit.merge(mechanics[projection], on="candidate_id", how="left", validate="one_to_one")
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
    return d

# ---------------------------------------------------------------------------
# Targeted surface-quality reread
# ---------------------------------------------------------------------------

def surface_metrics_for_selected_ts(
    audit: pd.DataFrame,
    canonical_root: Path,
    helper_script: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mod = load_module(helper_script, "phase7m11a_helper")
    if str(getattr(mod, "SCRIPT_VERSION", "")) != EXPECTED_7M11A_VERSION:
        raise RuntimeError(
            f"07m11a helper must be v{EXPECTED_7M11A_VERSION}; found {getattr(mod, 'SCRIPT_VERSION', None)}"
        )
    required = [
        "read_mark_rows_for_timestamps", "numericize_mark", "front_expiry_pair_metrics", "MARK_COLUMNS",
        "MarkTimestampCache", "nearest_prior_surface_timestamps",
    ]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"07m11a helper missing required targeted-surface functions: {missing}")

    quality_rows: list[dict[str, Any]] = []
    day_diag: list[dict[str, Any]] = []
    ts_cache = mod.MarkTimestampCache(canonical_root, max_days=3)

    for day_text, g in audit.groupby("candidate_date", sort=True):
        day = datetime.fromisoformat(str(day_text)).date()

        # IMPORTANT: do not reuse 7M.11A's persisted selected-surface timestamp
        # columns here.  Those columns can contain missing values and therefore
        # may have been serialized through float64; nanosecond timestamps (~1e18)
        # cannot be round-tripped exactly through float64.  Re-resolve the exact
        # latest-prior surface timestamp from canonical int64 ts_recv_ns values
        # using the original causal targets instead.
        targets: list[int] = []
        for r in g.itertuples(index=False):
            pre_target = int(r.pre_event_cutoff_ns)
            targets.extend([pre_target, pre_target - 1800 * NS_PER_SEC])

        all_ts, _cur_ts, _cur_rows = ts_cache.load_window(day)
        resolved = mod.nearest_prior_surface_timestamps(all_ts, targets)
        needed = [int(v) for v in resolved.values() if v is not None]

        mark = mod.read_mark_rows_for_timestamps(
            canonical_root,
            [day - timedelta(days=1), day],
            needed,
            mod.MARK_COLUMNS,
        )
        mark = mod.numericize_mark(mark) if not mark.empty else mark
        groups = {int(ts): f.copy() for ts, f in mark.groupby("ts_recv_ns", sort=False)} if not mark.empty else {}

        metric_cache: dict[tuple[int, int], dict[str, Any]] = {}

        def metrics_at_target(target_ns: int) -> tuple[int | None, dict[str, Any]]:
            surface_ts = resolved.get(int(target_ns))
            if surface_ts is None:
                return None, {}
            ts = int(surface_ts)
            key = (ts, int(target_ns))
            if key not in metric_cache:
                metric_cache[key] = mod.front_expiry_pair_metrics(groups.get(ts, pd.DataFrame()), int(target_ns))
            return ts, metric_cache[key]

        for r in g.itertuples(index=False):
            pre_target = int(r.pre_event_cutoff_ns)
            old_target = pre_target - 1800 * NS_PER_SEC
            pre_surface_ts, pre_m = metrics_at_target(pre_target)
            old_surface_ts, old_m = metrics_at_target(old_target)

            pre_front = pre_m.get("front_expiry_code")
            old_front = old_m.get("front_expiry_code")
            pre_second = pre_m.get("second_expiry_code")
            old_second = old_m.get("second_expiry_code")
            pre_strike = safe_float(pre_m.get("front_atm_strike"))
            old_strike = safe_float(old_m.get("front_atm_strike"))

            row = {
                "candidate_id": str(r.candidate_id),
                "candidate_date": str(r.candidate_date),
                "pre_surface_found": bool(pre_m),
                "old_surface_found": bool(old_m),
                "pre_surface_ts_recv_ns_exact": pre_surface_ts,
                "old_surface_ts_recv_ns_exact": old_surface_ts,
                "quality_front_expiry_same_raw": bool(pre_front and old_front and pre_front == old_front),
                "quality_second_expiry_same_raw": bool(pre_second and old_second and pre_second == old_second),
                "quality_front_atm_strike_same_raw": bool(
                    math.isfinite(pre_strike) and math.isfinite(old_strike) and pre_strike == old_strike
                ),
                "pre_25d_call_delta_abs_error_raw": safe_float(pre_m.get("front_25d_call_delta_abs_error")),
                "pre_25d_put_delta_abs_error_raw": safe_float(pre_m.get("front_25d_put_delta_abs_error")),
                "old_25d_call_delta_abs_error_raw": safe_float(old_m.get("front_25d_call_delta_abs_error")),
                "old_25d_put_delta_abs_error_raw": safe_float(old_m.get("front_25d_put_delta_abs_error")),
                "pre_front_expiry_code_raw": pre_front,
                "old_front_expiry_code_raw": old_front,
                "pre_second_expiry_code_raw": pre_second,
                "old_second_expiry_code_raw": old_second,
                "pre_front_atm_strike_raw": pre_strike,
                "old_front_atm_strike_raw": old_strike,
            }
            quality_rows.append(row)

        day_diag.append({
            "candidate_date": str(day_text),
            "candidates": int(len(g)),
            "causal_target_timestamps": int(len(set(targets))),
            "needed_surface_timestamps": int(len(set(needed))),
            "resolved_surface_timestamps": int(len({int(v) for v in resolved.values() if v is not None})),
            "loaded_mark_rows": int(len(mark)),
            "loaded_surface_timestamps": int(mark["ts_recv_ns"].nunique()) if not mark.empty else 0,
        })

    quality = pd.DataFrame(quality_rows)
    if len(quality) != len(audit) or quality["candidate_id"].duplicated().any():
        raise RuntimeError("Targeted options quality reread did not return exactly one row per 7M.11A candidate")
    return quality, pd.DataFrame(day_diag)

# ---------------------------------------------------------------------------
# Exposure freeze
# ---------------------------------------------------------------------------

def freeze_exposures(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = d.copy()

    # PRIMARY: direction-aligned approximate 25d skew change.
    skew_change = pd.to_numeric(x["anticipation_30m__front_25d_call_put_iv_diff_change"], errors="coerce")
    direction = pd.to_numeric(x["direction"], errors="coerce")
    directional_skew_change = direction * skew_change
    x["options_directional_25d_skew_change_30m"] = directional_skew_change

    err_cols = [
        "pre_25d_call_delta_abs_error_raw", "pre_25d_put_delta_abs_error_raw",
        "old_25d_call_delta_abs_error_raw", "old_25d_put_delta_abs_error_raw",
    ]
    skew_quality = bool_series(x["quality_front_expiry_same_raw"])
    for c in err_cols:
        v = pd.to_numeric(x[c], errors="coerce")
        skew_quality &= v.notna() & (v <= DELTA_TARGET_MAX_ABS_ERROR + EPS)
    x["quality_skew_25d_30m"] = skew_quality
    x["exp_options_directional_skew_change_30m"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[skew_quality & (directional_skew_change > EPS), "exp_options_directional_skew_change_30m"] = 1
    x.loc[skew_quality & (directional_skew_change < -EPS), "exp_options_directional_skew_change_30m"] = 0

    # SECONDARY: ATM IV expansion/contraction with exact contract comparability.
    atm_change = pd.to_numeric(x["anticipation_30m__front_atm_iv_change"], errors="coerce")
    atm_quality = bool_series(x["quality_front_expiry_same_raw"]) & bool_series(x["quality_front_atm_strike_same_raw"])
    x["quality_atm_iv_30m"] = atm_quality
    x["exp_options_atm_iv_change_30m"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[atm_quality & (atm_change > EPS), "exp_options_atm_iv_change_30m"] = 1
    x.loc[atm_quality & (atm_change < -EPS), "exp_options_atm_iv_change_30m"] = 0

    # SECONDARY: orient term change so positive means front-end richer vs second expiry.
    second_minus_front_change = pd.to_numeric(x["anticipation_30m__second_minus_front_atm_iv_change"], errors="coerce")
    front_end_richening = -second_minus_front_change
    x["options_front_end_richening_30m"] = front_end_richening
    term_quality = bool_series(x["quality_front_expiry_same_raw"]) & bool_series(x["quality_second_expiry_same_raw"])
    x["quality_term_structure_30m"] = term_quality
    x["exp_options_front_end_term_richening_30m"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[term_quality & (front_end_richening > EPS), "exp_options_front_end_term_richening_30m"] = 1
    x.loc[term_quality & (front_end_richening < -EPS), "exp_options_front_end_term_richening_30m"] = 0

    definitions = pd.DataFrame([
        {
            "family": "OPTIONS_DIRECTIONAL_25D_SKEW_CHANGE_30M",
            "priority": "PRIMARY",
            "exposure_column": "exp_options_directional_skew_change_30m",
            "high_state": "DIRECTIONAL_SKEW_ALIGNED",
            "low_state": "DIRECTIONAL_SKEW_COUNTER",
            "definition": "direction * 30m change(call25d IV - put25d IV) >0 vs <0; pre-event window ends before candidate_second",
            "quality_guard": f"front expiry identical; call/put delta abs error <= {DELTA_TARGET_MAX_ABS_ERROR:.2f} at BOTH endpoints; exact zero excluded",
            "boundary": 0.0,
            "boundary_basis": "physical direction-alignment sign boundary",
        },
        {
            "family": "OPTIONS_ATM_IV_CHANGE_30M",
            "priority": "SECONDARY",
            "exposure_column": "exp_options_atm_iv_change_30m",
            "high_state": "ATM_IV_EXPANSION",
            "low_state": "ATM_IV_CONTRACTION",
            "definition": "30m change in front-expiry ATM IV >0 vs <0; pre-event window ends before candidate_second",
            "quality_guard": "front expiry and exact selected ATM strike identical across both endpoints; exact zero excluded",
            "boundary": 0.0,
            "boundary_basis": "physical IV-change sign boundary",
        },
        {
            "family": "OPTIONS_FRONT_END_TERM_RICHENING_30M",
            "priority": "SECONDARY",
            "exposure_column": "exp_options_front_end_term_richening_30m",
            "high_state": "FRONT_END_RICHENING",
            "low_state": "FRONT_END_CHEAPENING",
            "definition": "negative 30m change(second-expiry ATM IV - front-expiry ATM IV) >0 vs <0",
            "quality_guard": "exact front and second expiry identities identical across both endpoints; exact zero excluded",
            "boundary": 0.0,
            "boundary_basis": "physical front-vs-next-expiry repricing sign boundary",
        },
    ])
    return x, definitions

# ---------------------------------------------------------------------------
# Matching
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


def direct_binary_pairs(d: pd.DataFrame, exposure: str, controls: Sequence[str]) -> pd.DataFrame:
    x = d.copy()
    exp = pd.to_numeric(x[exposure], errors="coerce")
    vals = set(exp.dropna().astype(int).unique().tolist())
    if not vals.issubset({0, 1}):
        raise RuntimeError(f"Non-binary exposure values for {exposure}: {sorted(vals)}")
    for c in controls:
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
                for c in controls:
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
                "direction": int(direction),
                "high_candidate_id": str(hr["candidate_id"]),
                "low_candidate_id": str(lr["candidate_id"]),
                "distance": float(best[0][0]),
            }
            for c, diff in zip(controls, diffs):
                row[f"high__{c}"] = safe_float(hr[c])
                row[f"low__{c}"] = safe_float(lr[c])
                row[f"diff__{c}"] = float(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def pair_balance(pairs: pd.DataFrame, controls: Sequence[str]) -> tuple[float, dict[str, float]]:
    vals: dict[str, float] = {}
    for c in controls:
        vals[c] = smd(
            pairs.get(f"high__{c}", pd.Series(dtype=float)),
            pairs.get(f"low__{c}", pd.Series(dtype=float)),
        )
    finite = [abs(v) for v in vals.values() if math.isfinite(v)]
    return (max(finite) if finite else math.nan), vals


def repair_pairs(pairs: pd.DataFrame, controls: Sequence[str]) -> pd.DataFrame:
    x = pairs.copy().reset_index(drop=True)
    while len(x) > MIN_REPAIR_PAIRS:
        mx, vals = pair_balance(x, controls)
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


def family_eligible(d: pd.DataFrame, exposure: str, controls: Sequence[str]) -> pd.DataFrame:
    z = d[bool_series(d["base_mechanics_eligible"])].copy()
    z = z[pd.to_numeric(z[exposure], errors="coerce").notna()].copy()
    for c in controls:
        z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()
    return z


def evaluate_family(
    d: pd.DataFrame,
    family: str,
    priority: str,
    exposure: str,
    controls: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    z = family_eligible(d, exposure, controls)
    exp = pd.to_numeric(z[exposure], errors="coerce").astype(int)
    high_n = int((exp == 1).sum())
    low_n = int((exp == 0).sum())

    pre = direct_binary_pairs(z, exposure, controls)
    post = repair_pairs(pre, controls) if len(pre) else pre
    mx, smds = pair_balance(post, controls) if len(post) else (math.nan, {})
    disc = int((post["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(post) else 0
    val = int((post["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(post) else 0
    feasible = bool(
        len(post) >= MIN_FULL_PAIRS
        and disc >= MIN_SPLIT_PAIRS
        and val >= MIN_SPLIT_PAIRS
        and math.isfinite(mx)
        and mx <= BALANCE_SMD_GATE
    )

    row: dict[str, Any] = {
        "family": family,
        "priority": priority,
        "exposure_column": exposure,
        "controls": ";".join(controls),
        "eligible_candidates": int(len(z)),
        "high_candidates": high_n,
        "low_candidates": low_n,
        "high_share": float(high_n / len(z)) if len(z) else math.nan,
        "low_share": float(low_n / len(z)) if len(z) else math.nan,
        "pairs_before_repair": int(len(pre)),
        "pairs_after_repair": int(len(post)),
        "discovery_pairs": disc,
        "validation_pairs": val,
        "max_abs_smd": mx,
        "full_pair_gate": MIN_FULL_PAIRS,
        "split_pair_gate": MIN_SPLIT_PAIRS,
        "balance_gate": BALANCE_SMD_GATE,
        "feasible_before_outcomes": feasible,
    }
    for c, v in smds.items():
        row[f"smd__{c}"] = v

    if len(post):
        post = post.copy()
        post.insert(0, "family", family)
        post.insert(1, "priority", priority)
        post.insert(2, "exposure_column", exposure)
        post.insert(3, "control_columns", ";".join(controls))
    return row, post

# ---------------------------------------------------------------------------
# Diagnostics / reports
# ---------------------------------------------------------------------------

def subgroup_rows(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in FAMILY_SPECS:
        z = family_eligible(d, spec["exposure"], spec["controls"])
        groups = [
            ("FULL", pd.Series(True, index=z.index)),
            ("DISCOVERY", z["candidate_date"].astype(str) <= DISCOVERY_END_DATE),
            ("VALIDATION", z["candidate_date"].astype(str) >= VALIDATION_START_DATE),
            ("UP", pd.to_numeric(z["direction"], errors="coerce") == 1),
            ("DOWN", pd.to_numeric(z["direction"], errors="coerce") == -1),
        ]
        for name, mask in groups:
            q = z[mask].copy()
            qe = pd.to_numeric(q[spec["exposure"]], errors="coerce")
            hi = int((qe == 1).sum())
            lo = int((qe == 0).sum())
            rows.append({
                "family": spec["family"],
                "priority": spec["priority"],
                "subgroup": name,
                "eligible_candidates": int(len(q)),
                "high_candidates": hi,
                "low_candidates": lo,
                "high_share": float(hi / len(q)) if len(q) else math.nan,
                "both_exposure_states_present": bool(hi > 0 and lo > 0),
            })
    return pd.DataFrame(rows)


def quality_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "25d_skew_quality": bool_series(d["quality_skew_25d_30m"]),
        "atm_iv_quality": bool_series(d["quality_atm_iv_30m"]),
        "term_structure_quality": bool_series(d["quality_term_structure_30m"]),
        "front_expiry_same_raw": bool_series(d["quality_front_expiry_same_raw"]),
        "second_expiry_same_raw": bool_series(d["quality_second_expiry_same_raw"]),
        "front_atm_strike_same_raw": bool_series(d["quality_front_atm_strike_same_raw"]),
    }
    for name, mask in masks.items():
        rows.append({"quality_guard": name, "passing_candidates": int(mask.sum()), "passing_pct": float(mask.mean() * 100.0)})
    for c in [
        "pre_25d_call_delta_abs_error_raw", "pre_25d_put_delta_abs_error_raw",
        "old_25d_call_delta_abs_error_raw", "old_25d_put_delta_abs_error_raw",
    ]:
        v = pd.to_numeric(d[c], errors="coerce")
        rows.append({
            "quality_guard": c,
            "passing_candidates": int((v <= DELTA_TARGET_MAX_ABS_ERROR + EPS).sum()),
            "passing_pct": float((v <= DELTA_TARGET_MAX_ABS_ERROR + EPS).mean() * 100.0),
        })
    return pd.DataFrame(rows)


def make_findings(gate: pd.DataFrame) -> str:
    lines = [
        "# Phase 7M.11B - BTC-options anticipation matching feasibility",
        "",
        "Status: **OUTCOME-BLIND MATCHING FEASIBILITY COMPLETE**",
        "",
        "Future outcome tables opened: **NO**",
        "Composite options signal built: **NO**",
        "OI/gamma hedging-pressure variables used: **NO**",
        "Event-time option response used: **NO**",
        "",
        "## Frozen hypotheses",
        "",
        "- **PRIMARY — OPTIONS_DIRECTIONAL_25D_SKEW_CHANGE_30M**: pre-event 30m approximate 25d skew repricing aligned vs counter to the upcoming perp direction, with a <=0.10 delta-error guard at both endpoints.",
        "- **SECONDARY — OPTIONS_ATM_IV_CHANGE_30M**: comparable-contract front ATM IV expansion vs contraction over the pre-event 30m window.",
        "- **SECONDARY — OPTIONS_FRONT_END_TERM_RICHENING_30M**: front expiry richening vs cheapening relative to the second expiry over the pre-event 30m window.",
        "",
        "## Family gates",
        "",
    ]
    for r in gate.itertuples(index=False):
        lines.append(
            f"- **{r.family} ({r.priority})**: eligible={int(r.eligible_candidates):,}, "
            f"high/low={int(r.high_candidates):,}/{int(r.low_candidates):,}, "
            f"pairs={int(r.pairs_after_repair):,}, D/V={int(r.discovery_pairs):,}/{int(r.validation_pairs):,}, "
            f"max |SMD|={safe_float(r.max_abs_smd):.4f}, PASS={bool(r.feasible_before_outcomes)}"
        )
    lines += [
        "",
        "## Quality contract",
        "",
        f"- Approximate 25d call and put selections must each be within {DELTA_TARGET_MAX_ABS_ERROR:.2f} delta of +/-0.25 at both endpoints.",
        "- ATM-IV change requires the same front expiry and exact same selected ATM strike across the 30m comparison.",
        "- Term-structure change requires the same front and second expiry identities across the 30m comparison.",
        "- These checks are recomputed from exact latest-prior canonical surfaces resolved from the original causal targets; persisted float-rounded surface timestamps are not trusted. Whole option days are not materialized.",
        "",
        "## Matching contract",
        "",
        "Each family is matched independently using exact candidate_date + direction and the frozen five causal perp controls. No options-family result can alter another family's pair set.",
        "",
        "A family that fails this gate remains **NOT TESTED** for outcomes under this design; thresholds and quality guards must not be relaxed after seeing this report.",
    ]
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    # Nanosecond timestamps must never be round-tripped through float64 when
    # exact equality against canonical ts_recv_ns is required.
    exact_ns = 1_768_000_000_123_456_789
    assert int(float(exact_ns)) != exact_ns

    d = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(8)],
        "candidate_date": ["2026-06-01"] * 8,
        "direction": [1, -1, 1, -1, 1, -1, 1, -1],
        "base_mechanics_eligible": [True] * 8,
        "anticipation_30m__front_25d_call_put_iv_diff_change": [0.02, -0.03, 0.0, 0.02, -0.01, 0.01, -0.02, 0.03],
        "anticipation_30m__front_atm_iv_change": [0.02, -0.01, 0.0, 0.03, -0.02, 0.01, -0.03, 0.04],
        "anticipation_30m__second_minus_front_atm_iv_change": [-0.02, 0.01, 0.0, -0.03, 0.02, -0.01, 0.03, -0.04],
        "quality_front_expiry_same_raw": [True] * 8,
        "quality_second_expiry_same_raw": [True] * 8,
        "quality_front_atm_strike_same_raw": [True] * 8,
        "pre_25d_call_delta_abs_error_raw": [0.05] * 8,
        "pre_25d_put_delta_abs_error_raw": [0.05] * 8,
        "old_25d_call_delta_abs_error_raw": [0.05, 0.05, 0.05, 0.11, 0.05, 0.05, 0.05, 0.05],
        "old_25d_put_delta_abs_error_raw": [0.05] * 8,
        "__momentum": np.tile([0.0, 0.1], 4),
        "__aggression": np.tile([1.0, 1.1], 4),
        "__destruction": np.tile([0.3, 0.31], 4),
        "__refill": np.tile([2.0, 2.1], 4),
        "__refill_breadth": np.tile([1.0, 1.1], 4),
    })
    x, defs = freeze_exposures(d)
    assert int(x.loc[0, "exp_options_directional_skew_change_30m"]) == 1
    assert int(x.loc[1, "exp_options_directional_skew_change_30m"]) == 1  # DOWN * negative skew change = aligned
    assert pd.isna(x.loc[2, "exp_options_directional_skew_change_30m"])
    assert pd.isna(x.loc[3, "exp_options_directional_skew_change_30m"])  # quality fail >0.10
    assert int(x.loc[0, "exp_options_atm_iv_change_30m"]) == 1
    assert int(x.loc[1, "exp_options_atm_iv_change_30m"]) == 0
    assert int(x.loc[0, "exp_options_front_end_term_richening_30m"]) == 1
    assert int(x.loc[1, "exp_options_front_end_term_richening_30m"]) == 0
    assert set(defs["priority"]) == {"PRIMARY", "SECONDARY"}

    rows = []
    for day in ["2026-06-10", "2026-06-20"]:
        for i in range(20):
            base = float(i % 5)
            for e in (0, 1):
                rows.append({
                    "candidate_id": f"{day}-{i}-{e}", "candidate_date": day, "direction": 1, "exp": e,
                    "__momentum": base, "__aggression": base + 1, "__destruction": base / 10,
                    "__refill": base + 2, "__refill_breadth": base + 3,
                })
    m = pd.DataFrame(rows)
    p = direct_binary_pairs(m, "exp", BASE_CONTROL_COLUMNS)
    assert len(p) == 40
    repaired = repair_pairs(p, BASE_CONTROL_COLUMNS)
    mx, _ = pair_balance(repaired, BASE_CONTROL_COLUMNS)
    assert len(repaired) == 40 and math.isfinite(mx) and mx <= 1e-12
    print(
        "SELF-TEST PASS: exact int64 surface timestamp discipline, 30m options anticipation sign/orientation, "
        "two-endpoint 25d delta-quality guards, ATM/term comparability guards, exact date+direction matching, "
        "and SMD repair logic validated."
    )
    return 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.11B outcome-blind BTC-options anticipation matching feasibility")
    p.add_argument("--audit-root", type=Path, default=DEFAULT_7M11A_ROOT)
    p.add_argument("--audit-report-root", type=Path, default=DEFAULT_7M11A_REPORT_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_7M6F_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--phase7m11a-script", type=Path, default=DEFAULT_7M11A_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_stack()

    print("Phase 7M.11B - Outcome-blind BTC-options anticipation matching feasibility")
    print("[1/7] Verifying frozen 7M.11A causal-only contract and loading substrates...")
    contract = verify_7m11a_contract(args.audit_root, args.audit_report_root)
    audit = load_7m11a(args.audit_root)
    mechanics = load_7m6f_primary(args.sequence_root)
    d = build_analysis_frame(audit, mechanics)

    print("[2/7] Re-auditing ONLY selected pre-event and pre-30m option surfaces for quality...")
    quality, quality_day_diag = surface_metrics_for_selected_ts(audit, args.canonical_root, args.phase7m11a_script)
    d = d.merge(quality, on=["candidate_id", "candidate_date"], how="left", validate="one_to_one")

    print("[3/7] Freezing 30m options anticipation exposures and outcome-blind quality guards...")
    d, definitions = freeze_exposures(d)
    quality_report = quality_summary(d)

    print("[4/7] Matching each options family independently under frozen causal controls...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    gate_rows: list[dict[str, Any]] = []
    pair_hashes: dict[str, str] = {}

    for spec in FAMILY_SPECS:
        row, pairs = evaluate_family(d, spec["family"], spec["priority"], spec["exposure"], spec["controls"])
        gate_rows.append(row)
        if len(pairs):
            slug = spec["family"].lower()
            csv_path = args.report_root / f"pair_manifest_{slug}.csv"
            pq_path = args.output_root / f"matched_pairs_{slug}_outcome_blind.parquet"
            atomic_csv(csv_path, pairs)
            atomic_parquet(pq_path, pairs)
            pair_hashes[spec["family"]] = sha256_file(pq_path)
        print(
            f"      {spec['family']} ({spec['priority']}): eligible={row['eligible_candidates']:,} "
            f"high/low={row['high_candidates']:,}/{row['low_candidates']:,} pairs={row['pairs_after_repair']:,} "
            f"D/V={row['discovery_pairs']:,}/{row['validation_pairs']:,} "
            f"max|SMD|={safe_float(row['max_abs_smd']):.4f} PASS={bool(row['feasible_before_outcomes'])}"
        )

    gate = pd.DataFrame(gate_rows)
    subgroups = subgroup_rows(d)

    print("[5/7] Writing outcome-blind pair manifests, quality diagnostics, and invariants...")
    atomic_csv(args.report_root / "family_matching_feasibility.csv", gate)
    atomic_csv(args.report_root / "exposure_definitions.csv", definitions)
    atomic_csv(args.report_root / "surface_quality_guard_summary.csv", quality_report)
    atomic_csv(args.report_root / "targeted_surface_read_by_day.csv", quality_day_diag)
    atomic_csv(args.report_root / "subgroup_representation.csv", subgroups)

    invariants = pd.DataFrame([
        {"invariant": "phase", "value": PHASE},
        {"invariant": "script_version", "value": SCRIPT_VERSION},
        {"invariant": "future_outcomes_opened", "value": False},
        {"invariant": "composite_signal_created", "value": False},
        {"invariant": "event_response_used", "value": False},
        {"invariant": "oi_gamma_used", "value": False},
        {"invariant": "option_trade_directional_semantics_used", "value": False},
        {"invariant": "pre_event_cutoff", "value": "candidate_second start boundary"},
        {"invariant": "anticipation_window_sec", "value": 1800},
        {"invariant": "delta_target_max_abs_error", "value": DELTA_TARGET_MAX_ABS_ERROR},
        {"invariant": "match_z_caliper", "value": MATCH_Z_CALIPER},
        {"invariant": "match_euclidean_caliper", "value": MATCH_EUCLIDEAN_CALIPER},
        {"invariant": "balance_smd_gate", "value": BALANCE_SMD_GATE},
        {"invariant": "min_full_pairs", "value": MIN_FULL_PAIRS},
        {"invariant": "min_split_pairs", "value": MIN_SPLIT_PAIRS},
        {"invariant": "source_7m11a_version", "value": EXPECTED_7M11A_VERSION},
        {"invariant": "quality_surface_timestamp_resolution", "value": "re_resolve_exact_int64_latest_prior_from_causal_targets"},
        {"invariant": "source_7m11a_causal_sha256", "value": contract["causal_sha256"]},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", invariants)
    atomic_text(args.report_root / "findings.md", make_findings(gate))

    print("[6/7] Persisting causal matching substrate and pair hashes...")
    causal_cols = [
        "candidate_id", "candidate_date", "period", "direction",
        "options_directional_25d_skew_change_30m", "options_front_end_richening_30m",
        "anticipation_30m__front_atm_iv_change",
        "quality_skew_25d_30m", "quality_atm_iv_30m", "quality_term_structure_30m",
        "exp_options_directional_skew_change_30m", "exp_options_atm_iv_change_30m",
        "exp_options_front_end_term_richening_30m",
        *BASE_CONTROL_COLUMNS,
    ]
    atomic_parquet(args.output_root / "candidate_options_anticipation_matching_substrate.parquet", d[causal_cols].copy())

    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "candidate_rows": int(len(d)),
        "future_outcomes_opened": False,
        "composite_signal_created": False,
        "primary_family": "OPTIONS_DIRECTIONAL_25D_SKEW_CHANGE_30M",
        "delta_target_max_abs_error": DELTA_TARGET_MAX_ABS_ERROR,
        "pair_hashes": pair_hashes,
        "families": gate.to_dict(orient="records"),
    }
    atomic_json(args.report_root / "summary.json", summary)

    print("[7/7] Complete.")
    for r in gate.itertuples(index=False):
        print(
            f"  {r.family}: pairs={int(r.pairs_after_repair):,} "
            f"D/V={int(r.discovery_pairs):,}/{int(r.validation_pairs):,} "
            f"max|SMD|={safe_float(r.max_abs_smd):.4f} PASS={bool(r.feasible_before_outcomes)}"
        )
    print(f"  reports={args.report_root}")
    print(f"  frozen pairs={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
