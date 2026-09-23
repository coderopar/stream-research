#!/usr/bin/env python3
"""Phase 7M.9B - outcome-blind Binance spot-confirmation matching feasibility.

Research question
-----------------
Can two causal Binance spot-confirmation primitives add information beyond the
already-characterized prospective 25-bp perp state while retaining enough
common support for a later outcome test?

Frozen exposures (measured causally at exact 7M.6F anchor+3s)
----------------------------------------------------------------
PRIMARY - SPOT_AGGRESSIVE_FLOW_30S
    high = direction-aligned net spot aggressive quote > 0
    low  = direction-aligned net spot aggressive quote < 0
    eligibility requires >=5 known-side spot aggTrades in (T-30s, T].
    Exact zero net flow is excluded.

    Additional family-specific control:
      log1p(30s total known-side spot aggressive quote)
    This holds overall spot execution activity roughly comparable while asking
    whether the *direction* of spot execution relative to the perp move matters.

SECONDARY - SPOT_PRICE_CONFIRMATION_5S
    high = direction-aligned strict 5s spot return > 0
    low  = direction-aligned strict 5s spot return < 0
    Both price endpoints must satisfy the frozen 7M.9A v1.0.1 rule: latest
    causal spot trade in (target-1s, target]. Exact zero return is excluded.

    Additional family-specific control:
      absolute 5s spot return magnitude
    This holds spot move magnitude roughly comparable while asking whether its
    sign agrees with the perp candidate.

Frozen base causal controls
---------------------------
Exact candidate_date + direction, then 1:1 no-replacement matching on:
* 3s directional perp momentum
* log1p(3s directional perp aggressive quote)
* initial opposing queue destruction
* log1p(anchor-relative 3s opposing refill quote)
* log1p(anchor-relative 3s opposing refill breadth)

Matching constants are retained from the 7M matching lineage:
* per-control |z difference| <= 0.75
* Euclidean z distance <= 1.00
* repair until max |SMD| <= 0.20
* full pairs >= 250
* discovery pairs >= 100
* validation pairs >= 100

Scientific safeguards
---------------------
* 7M.9A corrected causal contract is verified before matching.
* No future outcome file is opened.
* No raw depth replay.
* Spot LocalBook snapshots are not used.
* No composite spot-confirmation score.
* Primary and secondary families are matched independently.
* 30s spot-price sign is intentionally NOT tested (insufficient counter-state
  representation in 7M.9A).
* No caliper/threshold may be relaxed to force feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
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
PHASE = "7M.9B"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATES = 7067
EXPECTED_ANCHORED_3S = 7031
PRIMARY_HORIZON_SEC = 3
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20
MIN_FULL_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_KNOWN_SIDE_TRADES_FLOW = 5
PRICE_ENDPOINT_BIN_SEC = 1
MIN_REPAIR_PAIRS = 8
EPS = 1e-12

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M9A_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_binance_spot_confirmation_feasibility_v1"
)
DEFAULT_7M9A_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_binance_spot_confirmation_feasibility"
)
DEFAULT_7M6F_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_refill_redepletion_sequence_v2"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_binance_spot_confirmation_matching_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_binance_spot_confirmation_matching_feasibility"
)

AUDIT_COLUMNS = [
    "candidate_id", "candidate_date", "direction", "anchor_valid_3s",
    "decision_ts_recv_ns",
    "spot_directional_price_return_bps_5s",
    "spot_price_now_endpoint_bin_valid_5s",
    "spot_price_prior_endpoint_bin_valid_5s",
    "spot_directional_net_aggressive_quote_30s",
    "spot_known_side_trade_count_30s",
    "spot_known_side_quote_30s",
]
MECHANICS_COLUMNS = [
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "anchor_found", "sequence_valid", "horizon_complete",
    "opposing__observed_refill_quote", "opposing__refilled_level_count",
    "control__directional_mid_move_bps_3s",
    "control__directional_aggressive_quote_3s",
    "control__opposing_peak_initial_queue_loss_share_3s",
    "is_reaccel_context",
]
FORBIDDEN_TOKENS = (
    "outcome", "future", "p75", "p50", "p100", "mfe", "mae",
    "first_passage", "reach_", "reaches_", "h300_", "h600_", "h900_",
    "h1800_", "continuation_target", "success",
)
BASE_CONTROL_COLUMNS = [
    "__momentum", "__aggression", "__destruction", "__refill", "__refill_breadth",
]
FLOW_CONTROL_COLUMNS = [*BASE_CONTROL_COLUMNS, "__spot_gross_activity_30s"]
PRICE_CONTROL_COLUMNS = [*BASE_CONTROL_COLUMNS, "__spot_abs_return_5s"]

FAMILY_SPECS = [
    {
        "family": "SPOT_AGGRESSIVE_FLOW_30S",
        "priority": "PRIMARY",
        "exposure": "exp_spot_flow_30s",
        "controls": FLOW_CONTROL_COLUMNS,
        "high_state": "SPOT_FLOW_ALIGNED",
        "low_state": "SPOT_FLOW_COUNTER",
    },
    {
        "family": "SPOT_PRICE_CONFIRMATION_5S",
        "priority": "SECONDARY",
        "exposure": "exp_spot_price_5s",
        "controls": PRICE_CONTROL_COLUMNS,
        "high_state": "SPOT_PRICE_ALIGNED",
        "low_state": "SPOT_PRICE_COUNTER",
    },
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.9B requires pandas, numpy and pyarrow in the project .venv.")


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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m9b-", suffix=path.suffix or ".tmp", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m9b-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m9b-", suffix=".parquet", dir=str(path.parent))
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


# ---------------------------------------------------------------------------
# Frozen 7M.9A causal-contract verification
# ---------------------------------------------------------------------------


def verify_7m9a_contract(audit_root: Path, report_root: Path) -> dict[str, Any]:
    causal_path = audit_root / "candidate_binance_spot_confirmation_audit.parquet"
    summary_path = report_root / "summary.json"
    invariants_path = report_root / "analysis_invariants.csv"
    if not causal_path.exists():
        raise FileNotFoundError(causal_path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing corrected 7M.9A summary: {summary_path}. Rerun 07m9a v1.0.1 before 7M.9B."
        )
    if not invariants_path.exists():
        raise FileNotFoundError(invariants_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "phase": summary.get("phase") == "7M.9A",
        "script_version": summary.get("script_version") == "1.0.1",
        "candidate_count": int(summary.get("candidate_count", -1)) == EXPECTED_CANDIDATES,
        "anchor_valid_3s_count": int(summary.get("anchor_valid_3s_count", -1)) == EXPECTED_ANCHORED_3S,
        "price_endpoint_bin_sec": int(summary.get("price_endpoint_bin_sec", -1)) == PRICE_ENDPOINT_BIN_SEC,
        "flow_primary_min_known_side_trades": int(summary.get("flow_primary_min_known_side_trades", -1)) == MIN_KNOWN_SIDE_TRADES_FLOW,
        "future_outcomes_opened": summary.get("future_outcomes_opened") is False,
        "matching_performed": summary.get("matching_performed") is False,
        "composite_signal_built": summary.get("composite_signal_built") is False,
        "spot_book_snapshots_used": summary.get("spot_book_snapshots_used") is False,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "7M.9A corrected causal contract verification failed for: " + ", ".join(failed)
        )

    inv = pd.read_csv(invariants_path)
    inv_map = {str(r["invariant"]): r.get("value", r.get("actual")) for _, r in inv.iterrows()}
    endpoint_rule = str(inv_map.get("spot_price_endpoint_rule", ""))
    if "(T-1s,T]" not in endpoint_rule.replace(" ", ""):
        # Accept exact v1.0.1 canonical spelling too; strip-space check above is intentionally strict.
        endpoint_rule_raw = str(inv_map.get("spot_price_endpoint_rule", ""))
        if endpoint_rule_raw != "latest_trade_in_(T-1s,T]_else_unavailable":
            raise RuntimeError(f"Unexpected 7M.9A endpoint rule: {endpoint_rule_raw!r}")
    if str(inv_map.get("spot_book_snapshots_used", "")).strip().lower() not in {"false", "0"}:
        raise RuntimeError("7M.9A contract indicates spot LocalBook snapshots were used; 7M.9B forbids them")

    return {
        "causal_path": causal_path,
        "causal_sha256": sha256_file(causal_path),
        "summary_sha256": sha256_file(summary_path),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Causal substrate loading
# ---------------------------------------------------------------------------


def load_7m9a(root: Path) -> pd.DataFrame:
    path = root / "candidate_binance_spot_confirmation_audit.parquet"
    d = read_columns(path, AUDIT_COLUMNS)
    d["candidate_id"] = d["candidate_id"].astype(str)
    d["candidate_date"] = d["candidate_date"].astype(str)
    d["direction"] = pd.to_numeric(d["direction"], errors="raise").astype(int)
    if len(d) != EXPECTED_CANDIDATES or d["candidate_id"].duplicated().any():
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} unique 7M.9A candidates; "
            f"found rows={len(d):,}, unique={d['candidate_id'].nunique():,}"
        )
    anchored = int(bool_series(d["anchor_valid_3s"]).sum())
    if anchored != EXPECTED_ANCHORED_3S:
        raise RuntimeError(f"Expected {EXPECTED_ANCHORED_3S:,} 7M.9A anchor+3s rows; found {anchored:,}")
    return d


def load_7m6f_primary(root: Path) -> pd.DataFrame:
    paths = sorted((root / "candidate_metrics").glob("candidate_date=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No 7M.6F candidate metrics under {root / 'candidate_metrics'}")
    parts: list[pd.DataFrame] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        missing = [c for c in MECHANICS_COLUMNS if c not in names]
        if missing:
            raise RuntimeError(f"{path} missing required 7M.6F causal columns: {missing}")
        f = pq.read_table(path, columns=MECHANICS_COLUMNS).to_pandas()
        f = f[pd.to_numeric(f["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
        parts.append(f)
    d = pd.concat(parts, ignore_index=True, sort=False)
    d["candidate_id"] = d["candidate_id"].astype(str)
    d["candidate_date"] = d["candidate_date"].astype(str)
    d["direction"] = pd.to_numeric(d["direction"], errors="raise").astype(int)
    if len(d) != EXPECTED_CANDIDATES or d["candidate_id"].duplicated().any():
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} unique primary 7M.6F rows; "
            f"found rows={len(d):,}, unique={d['candidate_id'].nunique():,}"
        )
    return d


def build_analysis_frame(audit: pd.DataFrame, mechanics: pd.DataFrame) -> pd.DataFrame:
    check = audit[["candidate_id", "candidate_date", "direction"]].merge(
        mechanics[["candidate_id", "candidate_date", "direction"]],
        on="candidate_id", how="inner", validate="one_to_one", suffixes=("_a", "_m"),
    )
    if len(check) != EXPECTED_CANDIDATES:
        raise RuntimeError("7M.9A and 7M.6F candidate universes do not align one-to-one")
    if not (check["candidate_date_a"].astype(str) == check["candidate_date_m"].astype(str)).all():
        raise RuntimeError("Candidate-date mismatch between 7M.9A and 7M.6F")
    if not (
        pd.to_numeric(check["direction_a"], errors="coerce")
        == pd.to_numeric(check["direction_m"], errors="coerce")
    ).all():
        raise RuntimeError("Direction mismatch between 7M.9A and 7M.6F")

    keep_mech = [
        c for c in MECHANICS_COLUMNS
        if c not in {"candidate_id", "candidate_date", "direction", "observation_horizon_sec"}
    ]
    projection = ["candidate_id", *keep_mech]
    if len(projection) != len(set(projection)):
        raise RuntimeError(f"Internal duplicate mechanics projection: {projection}")
    d = audit.merge(mechanics[projection], on="candidate_id", how="left", validate="one_to_one")

    d["base_mechanics_eligible"] = (
        bool_series(d["anchor_valid_3s"])
        & bool_series(d["anchor_found"])
        & bool_series(d["sequence_valid"])
        & bool_series(d["horizon_complete"])
    )

    d["__momentum"] = pd.to_numeric(d["control__directional_mid_move_bps_3s"], errors="coerce")
    d["__aggression"] = np.log1p(
        pd.to_numeric(d["control__directional_aggressive_quote_3s"], errors="coerce").clip(lower=0)
    )
    d["__destruction"] = pd.to_numeric(
        d["control__opposing_peak_initial_queue_loss_share_3s"], errors="coerce"
    )
    d["__refill"] = np.log1p(
        pd.to_numeric(d["opposing__observed_refill_quote"], errors="coerce").clip(lower=0)
    )
    d["__refill_breadth"] = np.log1p(
        pd.to_numeric(d["opposing__refilled_level_count"], errors="coerce").clip(lower=0)
    )
    d["__spot_gross_activity_30s"] = np.log1p(
        pd.to_numeric(d["spot_known_side_quote_30s"], errors="coerce").clip(lower=0)
    )
    d["__spot_abs_return_5s"] = pd.to_numeric(
        d["spot_directional_price_return_bps_5s"], errors="coerce"
    ).abs()
    return d


# ---------------------------------------------------------------------------
# Exposure freeze
# ---------------------------------------------------------------------------


def freeze_exposures(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = d.copy()

    # PRIMARY: actual observed spot executions within the causal 30s window.
    flow = pd.to_numeric(x["spot_directional_net_aggressive_quote_30s"], errors="coerce")
    known = pd.to_numeric(x["spot_known_side_trade_count_30s"], errors="coerce")
    flow_eligible = flow.notna() & known.notna() & (known >= MIN_KNOWN_SIDE_TRADES_FLOW)
    x["exp_spot_flow_30s"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[flow_eligible & (flow > EPS), "exp_spot_flow_30s"] = 1
    x.loc[flow_eligible & (flow < -EPS), "exp_spot_flow_30s"] = 0

    # SECONDARY: strict 1s-bin causal endpoints inherited from corrected 7M.9A.
    ret5 = pd.to_numeric(x["spot_directional_price_return_bps_5s"], errors="coerce")
    now_ok = pd.to_numeric(x["spot_price_now_endpoint_bin_valid_5s"], errors="coerce").fillna(0) > 0
    prior_ok = pd.to_numeric(x["spot_price_prior_endpoint_bin_valid_5s"], errors="coerce").fillna(0) > 0
    price_eligible = ret5.notna() & now_ok & prior_ok
    x["exp_spot_price_5s"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[price_eligible & (ret5 > EPS), "exp_spot_price_5s"] = 1
    x.loc[price_eligible & (ret5 < -EPS), "exp_spot_price_5s"] = 0

    definitions = pd.DataFrame([
        {
            "family": "SPOT_AGGRESSIVE_FLOW_30S",
            "priority": "PRIMARY",
            "exposure_column": "exp_spot_flow_30s",
            "high_state": "SPOT_FLOW_ALIGNED",
            "low_state": "SPOT_FLOW_COUNTER",
            "definition": "30s direction-aligned net spot aggressive quote > 0 vs < 0; >=5 known-side spot trades required; exact zero excluded",
            "family_specific_control": "log1p(30s total known-side spot aggressive quote)",
            "boundary": 0.0,
            "boundary_basis": "physical sign boundary",
        },
        {
            "family": "SPOT_PRICE_CONFIRMATION_5S",
            "priority": "SECONDARY",
            "exposure_column": "exp_spot_price_5s",
            "high_state": "SPOT_PRICE_ALIGNED",
            "low_state": "SPOT_PRICE_COUNTER",
            "definition": "strict direction-aligned 5s spot return > 0 vs < 0; both 1s endpoint bins required; exact zero excluded",
            "family_specific_control": "absolute 5s spot return magnitude",
            "boundary": 0.0,
            "boundary_basis": "physical sign boundary",
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
    high_share = float(high_n / len(z)) if len(z) else math.nan

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
        "high_share": high_share,
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
# Diagnostics/reporting
# ---------------------------------------------------------------------------


def subgroup_rows(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in FAMILY_SPECS:
        z = family_eligible(d, spec["exposure"], spec["controls"])
        exp = pd.to_numeric(z[spec["exposure"]], errors="coerce")
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


def make_findings(gate: pd.DataFrame, definitions: pd.DataFrame) -> str:
    lines = [
        "# Phase 7M.9B - Binance spot-confirmation matching feasibility",
        "",
        "Status: **OUTCOME-BLIND MATCHING FEASIBILITY COMPLETE**",
        "",
        "Future outcome tables opened: **NO**",
        "Spot LocalBook snapshots used: **NO**",
        "Raw depth replay performed: **NO**",
        "Composite spot-confirmation signal built: **NO**",
        "",
        "## Frozen hypotheses",
        "",
        "- **PRIMARY — SPOT_AGGRESSIVE_FLOW_30S**: direction-aligned vs counter 30s net spot aggressive execution flow, with >=5 known-side trades and gross 30s spot activity additionally controlled.",
        "- **SECONDARY — SPOT_PRICE_CONFIRMATION_5S**: strict 5s spot price move aligned vs counter to the perp candidate, with absolute 5s spot-return magnitude additionally controlled.",
        "- **30s spot-price sign is not tested** because 7M.9A found insufficient counter-state representation for the frozen D/V pair gate.",
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
        "## Matching contract",
        "",
        "Both families use exact candidate_date + direction and the frozen 7M base controls. The flow family additionally controls log 30s total known-side spot aggressive quote; the price family additionally controls absolute 5s spot return magnitude.",
        "",
        "## Integrity constraints",
        "",
        "- 7M.9A v1.0.1 strict one-second endpoint bins are required for the 5s price family.",
        "- Spot flow is computed from trades actually observed in (T-30s,T]; no price forward-fill is involved.",
        "- Spot LocalBook snapshots remain deferred/not used because their integrity has not been independently established.",
        "- A family that fails this gate remains NOT TESTED for outcomes under this design; thresholds must not be relaxed after seeing this report.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> int:
    d = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(8)],
        "candidate_date": ["2026-06-01"] * 8,
        "direction": [1] * 8,
        "base_mechanics_eligible": [True] * 8,
        "spot_directional_net_aggressive_quote_30s": [10, -10, 0, 5, -5, 9, -9, 3],
        "spot_known_side_trade_count_30s": [10, 10, 10, 4, 5, 6, 7, 8],
        "spot_known_side_quote_30s": [100, 100, 100, 100, 90, 120, 110, 80],
        "spot_directional_price_return_bps_5s": [2, -2, 0, 1, -1, 3, -3, 4],
        "spot_price_now_endpoint_bin_valid_5s": [1, 1, 1, 1, 1, 0, 1, 1],
        "spot_price_prior_endpoint_bin_valid_5s": [1, 1, 1, 1, 1, 1, 0, 1],
        "__momentum": np.tile([0.0, 0.1], 4),
        "__aggression": np.tile([1.0, 1.1], 4),
        "__destruction": np.tile([0.3, 0.31], 4),
        "__refill": np.tile([2.0, 2.1], 4),
        "__refill_breadth": np.tile([1.0, 1.1], 4),
        "__spot_gross_activity_30s": np.log1p([100, 100, 100, 100, 90, 120, 110, 80]),
        "__spot_abs_return_5s": [2, 2, 0, 1, 1, 3, 3, 4],
    })
    x, defs = freeze_exposures(d)
    assert int(x.loc[0, "exp_spot_flow_30s"]) == 1
    assert int(x.loc[1, "exp_spot_flow_30s"]) == 0
    assert pd.isna(x.loc[2, "exp_spot_flow_30s"])  # exact zero excluded
    assert pd.isna(x.loc[3, "exp_spot_flow_30s"])  # only four known-side trades
    assert int(x.loc[4, "exp_spot_flow_30s"]) == 0
    assert int(x.loc[0, "exp_spot_price_5s"]) == 1
    assert int(x.loc[1, "exp_spot_price_5s"]) == 0
    assert pd.isna(x.loc[2, "exp_spot_price_5s"])  # exact zero excluded
    assert pd.isna(x.loc[5, "exp_spot_price_5s"])  # current endpoint invalid
    assert pd.isna(x.loc[6, "exp_spot_price_5s"])  # prior endpoint invalid
    assert set(defs["priority"]) == {"PRIMARY", "SECONDARY"}

    # Family-specific controls must be frozen and distinct.
    assert "__spot_gross_activity_30s" in FLOW_CONTROL_COLUMNS
    assert "__spot_abs_return_5s" in PRICE_CONTROL_COLUMNS
    assert "__spot_abs_return_5s" not in FLOW_CONTROL_COLUMNS
    assert "__spot_gross_activity_30s" not in PRICE_CONTROL_COLUMNS

    # Direct matcher: enough exactly balanced observations across D/V.
    rows: list[dict[str, Any]] = []
    controls = FLOW_CONTROL_COLUMNS
    for day in ["2026-06-10", "2026-06-20"]:
        for i in range(20):
            base = float(i % 5)
            for e in (0, 1):
                row = {
                    "candidate_id": f"{day}-{i}-{e}", "candidate_date": day, "direction": 1,
                    "exp": e,
                    "__momentum": base, "__aggression": base + 1,
                    "__destruction": base / 10, "__refill": base + 2,
                    "__refill_breadth": base + 3,
                    "__spot_gross_activity_30s": base + 4,
                }
                rows.append(row)
    m = pd.DataFrame(rows)
    p = direct_binary_pairs(m, "exp", controls)
    assert len(p) == 40
    repaired = repair_pairs(p, controls)
    mx, _ = pair_balance(repaired, controls)
    assert len(repaired) == 40 and math.isfinite(mx) and mx <= 1e-12

    print("SELF-TEST PASS: corrected 7M.9A exposure semantics, family-specific spot controls, exact date+direction matching, and SMD repair logic validated.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.9B outcome-blind Binance spot-confirmation matching feasibility")
    p.add_argument("--audit-root", type=Path, default=DEFAULT_7M9A_ROOT)
    p.add_argument("--audit-report-root", type=Path, default=DEFAULT_7M9A_REPORT_ROOT)
    p.add_argument("--sequence-root", type=Path, default=DEFAULT_7M6F_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    require_stack()

    print("Phase 7M.9B - Outcome-blind Binance spot-confirmation matching feasibility")
    print("[1/6] Verifying corrected 7M.9A causal contract and loading frozen substrates...")
    contract = verify_7m9a_contract(args.audit_root, args.audit_report_root)
    audit = load_7m9a(args.audit_root)
    mechanics = load_7m6f_primary(args.sequence_root)
    d = build_analysis_frame(audit, mechanics)

    print("[2/6] Freezing primary spot-flow and secondary strict-price exposures...")
    d, definitions = freeze_exposures(d)

    print("[3/6] Matching each spot family independently under frozen causal controls...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    gate_rows: list[dict[str, Any]] = []
    pair_parts: list[pd.DataFrame] = []
    pair_hashes: dict[str, str] = {}
    for spec in FAMILY_SPECS:
        row, pairs = evaluate_family(
            d,
            spec["family"],
            spec["priority"],
            spec["exposure"],
            spec["controls"],
        )
        gate_rows.append(row)
        if len(pairs):
            slug = spec["family"].lower()
            csv_path = args.report_root / f"pair_manifest_{slug}.csv"
            pq_path = args.output_root / f"matched_pairs_{slug}_outcome_blind.parquet"
            atomic_csv(csv_path, pairs)
            atomic_parquet(pq_path, pairs)
            pair_hashes[spec["family"]] = sha256_file(pq_path)
            pair_parts.append(pairs)
        print(
            f"      {spec['family']} ({spec['priority']}): eligible={row['eligible_candidates']:,} "
            f"high/low={row['high_candidates']:,}/{row['low_candidates']:,} "
            f"pairs={row['pairs_after_repair']:,} D/V={row['discovery_pairs']:,}/{row['validation_pairs']:,} "
            f"max|SMD|={safe_float(row['max_abs_smd']):.4f} PASS={bool(row['feasible_before_outcomes'])}"
        )

    gate = pd.DataFrame(gate_rows)
    all_pairs = pd.concat(pair_parts, ignore_index=True, sort=False) if pair_parts else pd.DataFrame()

    print("[4/6] Auditing discovery/validation and direction representation...")
    subgroups = subgroup_rows(d)

    print("[5/6] Writing outcome-blind manifests, gates, and invariants...")
    atomic_csv(args.report_root / "exposure_definitions.csv", definitions)
    atomic_csv(args.report_root / "family_matching_feasibility.csv", gate)
    atomic_csv(args.report_root / "subgroup_representation.csv", subgroups)
    if len(all_pairs):
        atomic_csv(args.report_root / "all_family_pair_manifest.csv", all_pairs)

    candidate_cols = [
        "candidate_id", "candidate_date", "direction", "base_mechanics_eligible",
        "spot_directional_net_aggressive_quote_30s", "spot_known_side_trade_count_30s",
        "spot_known_side_quote_30s", "spot_directional_price_return_bps_5s",
        "spot_price_now_endpoint_bin_valid_5s", "spot_price_prior_endpoint_bin_valid_5s",
        "exp_spot_flow_30s", "exp_spot_price_5s",
        *BASE_CONTROL_COLUMNS, "__spot_gross_activity_30s", "__spot_abs_return_5s",
    ]
    assert_causal_names(candidate_cols, "7M.9B candidate manifest")
    atomic_parquet(
        args.output_root / "candidate_spot_exposure_manifest_outcome_blind.parquet",
        d[candidate_cols].copy(),
    )

    primary_pass = bool(
        gate.loc[gate["priority"] == "PRIMARY", "feasible_before_outcomes"].all()
    ) if (gate["priority"] == "PRIMARY").any() else False
    secondary_pass = bool(
        gate.loc[gate["priority"] == "SECONDARY", "feasible_before_outcomes"].all()
    ) if (gate["priority"] == "SECONDARY").any() else False
    all_pass = bool(gate["feasible_before_outcomes"].all()) if len(gate) else False

    inv = pd.DataFrame([
        {"invariant": "future_outcomes_opened", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_depth_replay_performed", "pass": True, "actual": False, "required": False},
        {"invariant": "spot_book_snapshots_used", "pass": True, "actual": False, "required": False},
        {"invariant": "composite_spot_signal_built", "pass": True, "actual": False, "required": False},
        {"invariant": "corrected_7m9a_version", "pass": True, "actual": "1.0.1", "required": "1.0.1"},
        {"invariant": "spot_price_endpoint_rule", "pass": True, "actual": "latest_trade_in_(T-1s,T]_else_unavailable", "required": "strict_1s_causal_bin"},
        {"invariant": "flow_min_known_side_trades", "pass": True, "actual": MIN_KNOWN_SIDE_TRADES_FLOW, "required": 5},
        {"invariant": "families_tested_independently", "pass": True, "actual": 2, "required": 2},
        {"invariant": "spot_price_30s_tested", "pass": True, "actual": False, "required": False},
        {"invariant": "exact_match_cells", "pass": True, "actual": "candidate_date+direction", "required": "candidate_date+direction"},
        {"invariant": "z_caliper", "pass": True, "actual": MATCH_Z_CALIPER, "required": 0.75},
        {"invariant": "euclidean_caliper", "pass": True, "actual": MATCH_EUCLIDEAN_CALIPER, "required": 1.00},
        {"invariant": "balance_gate", "pass": True, "actual": BALANCE_SMD_GATE, "required": 0.20},
        {"invariant": "full_pair_gate", "pass": True, "actual": MIN_FULL_PAIRS, "required": 250},
        {"invariant": "split_pair_gate", "pass": True, "actual": MIN_SPLIT_PAIRS, "required": 100},
        {"invariant": "primary_flow_controls_gross_spot_activity", "pass": True, "actual": "__spot_gross_activity_30s", "required": "yes"},
        {"invariant": "secondary_price_controls_abs_return", "pass": True, "actual": "__spot_abs_return_5s", "required": "yes"},
        {"invariant": "primary_family_gate_pass", "pass": primary_pass, "actual": primary_pass, "required": True},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)

    findings = make_findings(gate, definitions)
    atomic_text(args.report_root / "findings.md", findings)
    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "symbol": SYMBOL,
        "candidate_count": int(len(d)),
        "base_mechanics_eligible": int(bool_series(d["base_mechanics_eligible"]).sum()),
        "future_outcomes_opened": False,
        "raw_depth_replay_performed": False,
        "spot_book_snapshots_used": False,
        "composite_spot_signal_built": False,
        "source_7m9a": {
            "causal_sha256": contract["causal_sha256"],
            "summary_sha256": contract["summary_sha256"],
            "required_version": "1.0.1",
        },
        "matching": {
            "exact_cells": ["candidate_date", "direction"],
            "base_controls": BASE_CONTROL_COLUMNS,
            "flow_controls": FLOW_CONTROL_COLUMNS,
            "price_controls": PRICE_CONTROL_COLUMNS,
            "z_caliper": MATCH_Z_CALIPER,
            "euclidean_caliper": MATCH_EUCLIDEAN_CALIPER,
            "balance_smd_gate": BALANCE_SMD_GATE,
            "full_pair_gate": MIN_FULL_PAIRS,
            "split_pair_gate": MIN_SPLIT_PAIRS,
        },
        "families": gate.to_dict(orient="records"),
        "pair_manifest_sha256": pair_hashes,
        "primary_family_gate_pass": primary_pass,
        "secondary_family_gate_pass": secondary_pass,
        "all_family_gates_pass": all_pass,
        "next_step": "Review outcome-blind family gates. Only passing frozen pair manifests may proceed to a separately authorized P75 outcome phase.",
    }
    atomic_json(args.report_root / "summary.json", summary)

    print("[6/6] Complete.")
    print(f"      report={args.report_root}")
    print(f"      output={args.output_root}")
    print(f"      outcomes_opened=NO spot_book=NO primary_pass={primary_pass} secondary_pass={secondary_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
