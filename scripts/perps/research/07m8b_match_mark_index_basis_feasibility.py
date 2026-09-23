#!/usr/bin/env python3
"""Phase 7M.8B - outcome-blind matching feasibility for perp mark/index basis state.

Research question
-----------------
Do three causal mark/index/funding states add information beyond the already-
characterized 25-bp visible-book state while retaining enough common support
for a later outcome test?

Frozen exposures (all measured causally at exact 7M.6F anchor+3s)
-----------------------------------------------------------------
1. CURRENT_DIRECTIONAL_BASIS:
      direction-aligned current mark-index basis > 0 vs < 0.
   Exact zeros are excluded. Basis is 10,000 * (mark_price/index_price - 1).

2. DIRECTIONAL_BASIS_CHANGE_30S:
      direction-aligned 30s basis change > 0 vs < 0.
   Exact zeros are excluded. The 7M.8A audit showed the 30s causal span is
   effectively exact and has full usable coverage.

3. CURRENT_DIRECTIONAL_FUNDING:
      direction-aligned current funding rate > 0 vs < 0.
   Exact zeros are excluded. Funding is treated as a slower contract-level
   crowding/regime state, not as short-horizon flow.

Frozen matching state
---------------------
Exact candidate_date + direction, then 1:1 no-replacement matching on:
* 3s directional momentum
* log1p(3s directional aggressive quote)
* initial opposing queue destruction
* log1p(anchor-relative 3s opposing refill quote)
* log1p(anchor-relative 3s opposing refill breadth)

Matching constants are retained from Phase 7M.6B/7M.6E/7M.7B:
* per-control |z difference| <= 0.75
* Euclidean z distance <= 1.00
* repair until max |SMD| <= 0.20
* full pairs >= 250
* discovery pairs >= 100
* validation pairs >= 100
* both exposure states must each represent at least 10% of the retained
  binary exposure sample.

Scientific safeguards
---------------------
* No future outcome file is opened.
* No raw market replay.
* No composite signal.
* Each family is matched/tested independently.
* All three exposure boundaries are physical zero boundaries, fixed before
  outcomes. No terciles or outcome-facing cutoffs are selected here.
* No caliper or threshold is relaxed to force feasibility.
* Pair manifests are outcome-blind and may be frozen for a later outcome phase
  only after this feasibility report is reviewed.
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
PHASE = "7M.8B"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATES = 7067
PRIMARY_HORIZON_SEC = 3
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20
MIN_FULL_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_EXPOSURE_SHARE = 0.10
MAX_EXPOSURE_SHARE = 0.90
MIN_REPAIR_PAIRS = 8
EPS = 1e-12

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M8A_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_mark_index_basis_feasibility_v1"
)
DEFAULT_7M6F_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_refill_redepletion_sequence_v2"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "prospective_25bp_mark_index_basis_matching_v1"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_prospective_25bp_mark_index_basis_matching_feasibility"
)

# Explicit causal-only allow lists. Future labels are never requested.
AUDIT_COLUMNS = [
    "candidate_id", "candidate_date", "direction", "anchor_valid_3s",
    "decision_ts_recv_ns", "mark_event_found", "basis_valid",
    "directional_basis_bps",
    "w30s_prior_found", "w30s_directional_basis_change_bps",
    "funding_rate", "directional_funding_rate",
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

CONTROL_COLUMNS = [
    "__momentum", "__aggression", "__destruction", "__refill", "__refill_breadth",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.8B requires pandas, numpy and pyarrow in the project .venv.")


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
    bad = []
    for c in columns:
        cl = str(c).lower()
        if any(tok in cl for tok in FORBIDDEN_TOKENS):
            bad.append(str(c))
    if bad:
        raise RuntimeError(f"{context}: future/outcome-like columns are forbidden: {bad}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m8b-", suffix=path.suffix or ".tmp", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7m8b-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m8b-", suffix=".parquet", dir=str(path.parent))
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
# Causal substrate loading
# ---------------------------------------------------------------------------


def load_7m8a(root: Path) -> pd.DataFrame:
    path = root / "candidate_mark_index_basis_audit.parquet"
    d = read_columns(path, AUDIT_COLUMNS)
    d["candidate_id"] = d["candidate_id"].astype(str)
    d["candidate_date"] = d["candidate_date"].astype(str)
    d["direction"] = pd.to_numeric(d["direction"], errors="raise").astype(int)
    if len(d) != EXPECTED_CANDIDATES or d["candidate_id"].duplicated().any():
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} unique 7M.8A candidates; "
            f"found rows={len(d):,}, unique={d['candidate_id'].nunique():,}"
        )
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
    # Prevent accidental disagreement between the two frozen causal substrates.
    check = audit[["candidate_id", "candidate_date", "direction"]].merge(
        mechanics[["candidate_id", "candidate_date", "direction"]],
        on="candidate_id", how="inner", validate="one_to_one", suffixes=("_a", "_m"),
    )
    if len(check) != EXPECTED_CANDIDATES:
        raise RuntimeError("7M.8A and 7M.6F candidate universes do not align one-to-one")
    if not (check["candidate_date_a"].astype(str) == check["candidate_date_m"].astype(str)).all():
        raise RuntimeError("Candidate-date mismatch between 7M.8A and 7M.6F")
    if not (pd.to_numeric(check["direction_a"], errors="coerce") == pd.to_numeric(check["direction_m"], errors="coerce")).all():
        raise RuntimeError("Direction mismatch between 7M.8A and 7M.6F")

    keep_mech = [
        c for c in MECHANICS_COLUMNS
        if c not in {"candidate_id", "candidate_date", "direction", "observation_horizon_sec"}
    ]
    mechanics_projection = ["candidate_id", *keep_mech]
    if len(mechanics_projection) != len(set(mechanics_projection)):
        raise RuntimeError(f"Internal error: duplicate mechanics projection columns: {mechanics_projection}")
    d = audit.merge(
        mechanics[mechanics_projection],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )

    valid = (
        bool_series(d["anchor_valid_3s"])
        & bool_series(d["anchor_found"])
        & bool_series(d["sequence_valid"])
        & bool_series(d["horizon_complete"])
    )
    d["base_mechanics_eligible"] = valid

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
    return d


# ---------------------------------------------------------------------------
# Exposure freeze
# ---------------------------------------------------------------------------


def freeze_exposures(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = d.copy()

    # Current direction-aligned basis: physical zero sign boundary.
    basis = pd.to_numeric(x["directional_basis_bps"], errors="coerce")
    x["exp_current_directional_basis"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[basis > EPS, "exp_current_directional_basis"] = 1
    x.loc[basis < -EPS, "exp_current_directional_basis"] = 0

    # 30s direction-aligned basis change: physical zero sign boundary.
    basis_chg = pd.to_numeric(x["w30s_directional_basis_change_bps"], errors="coerce")
    prior_found = bool_series(x["w30s_prior_found"])
    x["exp_directional_basis_change_30s"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[prior_found & (basis_chg > EPS), "exp_directional_basis_change_30s"] = 1
    x.loc[prior_found & (basis_chg < -EPS), "exp_directional_basis_change_30s"] = 0

    # Current direction-aligned funding: physical zero sign boundary.
    funding = pd.to_numeric(x["directional_funding_rate"], errors="coerce")
    x["exp_current_directional_funding"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x.loc[funding > EPS, "exp_current_directional_funding"] = 1
    x.loc[funding < -EPS, "exp_current_directional_funding"] = 0

    definitions = pd.DataFrame([
        {
            "family": "CURRENT_DIRECTIONAL_BASIS",
            "exposure_column": "exp_current_directional_basis",
            "high_state": "BASIS_ALIGNED_WITH_MOVE",
            "low_state": "BASIS_COUNTER_TO_MOVE",
            "definition": "directional_basis_bps > 0 vs < 0; exact zero excluded",
            "lower_boundary": 0.0,
            "upper_boundary": 0.0,
            "boundary_basis": "physical sign boundary",
        },
        {
            "family": "DIRECTIONAL_BASIS_CHANGE_30S",
            "exposure_column": "exp_directional_basis_change_30s",
            "high_state": "BASIS_WIDENING_WITH_MOVE",
            "low_state": "BASIS_MOVING_AGAINST_MOVE",
            "definition": "w30s_directional_basis_change_bps > 0 vs < 0 with causal prior endpoint; exact zero excluded",
            "lower_boundary": 0.0,
            "upper_boundary": 0.0,
            "boundary_basis": "physical sign boundary",
        },
        {
            "family": "CURRENT_DIRECTIONAL_FUNDING",
            "exposure_column": "exp_current_directional_funding",
            "high_state": "FUNDING_ALIGNED_WITH_MOVE",
            "low_state": "FUNDING_COUNTER_TO_MOVE",
            "definition": "directional_funding_rate > 0 vs < 0; exact zero excluded",
            "lower_boundary": 0.0,
            "upper_boundary": 0.0,
            "boundary_basis": "physical sign boundary",
        },
    ])
    return x, definitions


# ---------------------------------------------------------------------------
# Matching and balance
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


def family_eligible(d: pd.DataFrame, exposure: str) -> pd.DataFrame:
    z = d[bool_series(d["base_mechanics_eligible"])].copy()
    z = z[pd.to_numeric(z[exposure], errors="coerce").notna()].copy()
    for c in CONTROL_COLUMNS:
        z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()
    return z


def evaluate_family(d: pd.DataFrame, family: str, exposure: str) -> tuple[dict[str, Any], pd.DataFrame]:
    z = family_eligible(d, exposure)
    exp = pd.to_numeric(z[exposure], errors="coerce").astype(int)
    high_n = int((exp == 1).sum())
    low_n = int((exp == 0).sum())
    high_share = float(high_n / len(z)) if len(z) else math.nan

    pre = direct_binary_pairs(z, exposure, CONTROL_COLUMNS)
    post = repair_pairs(pre, CONTROL_COLUMNS) if len(pre) else pre
    mx, smds = pair_balance(post, CONTROL_COLUMNS) if len(post) else (math.nan, {})
    disc = int((post["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(post) else 0
    val = int((post["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(post) else 0
    feasible = bool(
        len(post) >= MIN_FULL_PAIRS
        and disc >= MIN_SPLIT_PAIRS
        and val >= MIN_SPLIT_PAIRS
        and math.isfinite(mx)
        and mx <= BALANCE_SMD_GATE
        and math.isfinite(high_share)
        and MIN_EXPOSURE_SHARE <= high_share <= MAX_EXPOSURE_SHARE
    )

    row: dict[str, Any] = {
        "family": family,
        "exposure_column": exposure,
        "eligible_candidates": int(len(z)),
        "high_candidates": high_n,
        "low_candidates": low_n,
        "high_share": high_share,
        "pairs_before_repair": int(len(pre)),
        "pairs_after_repair": int(len(post)),
        "discovery_pairs": disc,
        "validation_pairs": val,
        "max_abs_smd": mx,
        "full_pair_gate": MIN_FULL_PAIRS,
        "split_pair_gate": MIN_SPLIT_PAIRS,
        "balance_gate": BALANCE_SMD_GATE,
        "min_exposure_share_gate": MIN_EXPOSURE_SHARE,
        "max_exposure_share_gate": MAX_EXPOSURE_SHARE,
        "feasible_before_outcomes": feasible,
    }
    for c, v in smds.items():
        row[f"smd__{c}"] = v

    if len(post):
        post = post.copy()
        post.insert(0, "family", family)
        post.insert(1, "exposure_column", exposure)
    return row, post


# ---------------------------------------------------------------------------
# Diagnostics/reporting
# ---------------------------------------------------------------------------


def subgroup_rows(d: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in definitions.itertuples(index=False):
        z = family_eligible(d, r.exposure_column)
        exp = pd.to_numeric(z[r.exposure_column], errors="coerce")
        groups = [
            ("FULL", pd.Series(True, index=z.index)),
            ("DISCOVERY", z["candidate_date"].astype(str) <= DISCOVERY_END_DATE),
            ("VALIDATION", z["candidate_date"].astype(str) >= VALIDATION_START_DATE),
            ("UP", pd.to_numeric(z["direction"], errors="coerce") == 1),
            ("DOWN", pd.to_numeric(z["direction"], errors="coerce") == -1),
        ]
        for name, mask in groups:
            q = z[mask].copy()
            qe = pd.to_numeric(q[r.exposure_column], errors="coerce")
            hi = int((qe == 1).sum())
            lo = int((qe == 0).sum())
            rows.append({
                "family": r.family,
                "subgroup": name,
                "eligible_candidates": int(len(q)),
                "high_candidates": hi,
                "low_candidates": lo,
                "high_share": float(hi / len(q)) if len(q) else math.nan,
                "both_exposure_states_present": bool(hi > 0 and lo > 0),
            })
    return pd.DataFrame(rows)


def make_findings(gate: pd.DataFrame, definitions: pd.DataFrame, subgroups: pd.DataFrame) -> str:
    lines = [
        "# Phase 7M.8B - mark/index-basis matching feasibility",
        "",
        "Status: **OUTCOME-BLIND MATCHING FEASIBILITY COMPLETE**",
        "",
        "Future outcome tables opened: **NO**",
        "Composite signal built: **NO**",
        "Raw market replay performed: **NO**",
        "",
        "## Frozen causal controls",
        "",
        "Exact candidate_date + direction, then 1:1 no-replacement matching on 3s momentum, log aggression, initial opposing destruction, log anchor-relative opposing refill amount, and log refill breadth.",
        "",
        "## Family gates",
        "",
    ]
    for r in gate.itertuples(index=False):
        lines.append(
            f"- **{r.family}**: eligible={int(r.eligible_candidates):,}, "
            f"high/low={int(r.high_candidates):,}/{int(r.low_candidates):,}, "
            f"pairs={int(r.pairs_after_repair):,}, D/V={int(r.discovery_pairs):,}/{int(r.validation_pairs):,}, "
            f"max |SMD|={safe_float(r.max_abs_smd):.4f}, PASS={bool(r.feasible_before_outcomes)}"
        )
    lines += ["", "## Exposure definitions", ""]
    for r in definitions.itertuples(index=False):
        lines.append(f"- `{r.family}`: {r.definition}")
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- Current mark-index basis is a perp-vs-reference dislocation state, not proof that the perpetual caused the move.",
        "- 30s basis change measures whether that dislocation widened with or against the candidate direction before the +3s decision point.",
        "- Funding is a slower contract-level crowding/regime state; it is not interpreted as short-horizon flow or participant identity.",
        "- All exposure boundaries are zero sign boundaries fixed without outcomes.",
        "",
        "A family that fails this gate must remain NOT TESTED for outcomes under this design; matching thresholds must not be loosened post hoc.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> int:
    d = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(12)],
        "candidate_date": ["2026-06-01"] * 12,
        "direction": [1] * 12,
        "base_mechanics_eligible": [True] * 12,
        "directional_basis_bps": [-2, -1, 0, 1, 2, -3, 3, -4, 4, -5, 5, 6],
        "w30s_prior_found": [True] * 12,
        "w30s_directional_basis_change_bps": [-.2, -.1, 0, .1, .2, -.3, .3, -.4, .4, -.5, .5, .6],
        "directional_funding_rate": [-2e-4, -1e-4, 0, 1e-4, 2e-4, -3e-4, 3e-4, -4e-4, 4e-4, -5e-4, 5e-4, 6e-4],
        "__momentum": np.tile([0.0, 0.1], 6),
        "__aggression": np.tile([1.0, 1.1], 6),
        "__destruction": np.tile([0.3, 0.31], 6),
        "__refill": np.tile([2.0, 2.1], 6),
        "__refill_breadth": np.tile([1.0, 1.1], 6),
    })
    mechanics_projection = [
        "candidate_id",
        *[c for c in MECHANICS_COLUMNS if c not in {"candidate_id", "candidate_date", "direction", "observation_horizon_sec"}],
    ]
    assert mechanics_projection.count("candidate_id") == 1
    assert len(mechanics_projection) == len(set(mechanics_projection))

    x, defs = freeze_exposures(d)
    assert int(x.loc[0, "exp_current_directional_basis"]) == 0
    assert int(x.loc[3, "exp_current_directional_basis"]) == 1
    assert pd.isna(x.loc[2, "exp_current_directional_basis"])
    assert int(x.loc[0, "exp_directional_basis_change_30s"]) == 0
    assert int(x.loc[3, "exp_directional_basis_change_30s"]) == 1
    assert pd.isna(x.loc[2, "exp_directional_basis_change_30s"])
    assert int(x.loc[0, "exp_current_directional_funding"]) == 0
    assert int(x.loc[3, "exp_current_directional_funding"]) == 1
    assert pd.isna(x.loc[2, "exp_current_directional_funding"])
    assert len(defs) == 3

    rows = []
    for day in ["2026-06-10", "2026-06-20"]:
        for i in range(20):
            base = float(i % 5)
            for e in (0, 1):
                rows.append({
                    "candidate_id": f"{day}-{i}-{e}", "candidate_date": day, "direction": 1,
                    "exp": e,
                    "__momentum": base, "__aggression": base + 1,
                    "__destruction": base / 10, "__refill": base + 2,
                    "__refill_breadth": base + 3,
                })
    m = pd.DataFrame(rows)
    p = direct_binary_pairs(m, "exp", CONTROL_COLUMNS)
    assert len(p) == 40
    repaired = repair_pairs(p, CONTROL_COLUMNS)
    mx, _ = pair_balance(repaired, CONTROL_COLUMNS)
    assert len(repaired) == 40 and math.isfinite(mx) and mx <= 1e-12
    print("SELF-TEST PASS: basis/funding zero-boundary exposure freeze, unique mechanics merge projection, exact date+direction binary matching, and SMD repair logic validated.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.8B outcome-blind mark/index-basis matching feasibility")
    p.add_argument("--audit-root", type=Path, default=DEFAULT_7M8A_ROOT)
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

    print("Phase 7M.8B - Outcome-blind mark/index-basis matching feasibility")
    print("[1/6] Loading frozen causal-only 7M.8A and 7M.6F substrates...")
    audit_path = args.audit_root / "candidate_mark_index_basis_audit.parquet"
    audit = load_7m8a(args.audit_root)
    mechanics = load_7m6f_primary(args.sequence_root)
    d = build_analysis_frame(audit, mechanics)

    print("[2/6] Freezing three zero-boundary basis/funding exposure definitions before outcomes...")
    d, definitions = freeze_exposures(d)

    print("[3/6] Matching each family independently under frozen 7M.6 controls...")
    family_specs = [
        ("CURRENT_DIRECTIONAL_BASIS", "exp_current_directional_basis"),
        ("DIRECTIONAL_BASIS_CHANGE_30S", "exp_directional_basis_change_30s"),
        ("CURRENT_DIRECTIONAL_FUNDING", "exp_current_directional_funding"),
    ]
    gate_rows: list[dict[str, Any]] = []
    pair_parts: list[pd.DataFrame] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    for family, exposure in family_specs:
        row, pairs = evaluate_family(d, family, exposure)
        gate_rows.append(row)
        if len(pairs):
            pair_parts.append(pairs)
            slug = family.lower()
            atomic_csv(args.report_root / f"pair_manifest_{slug}.csv", pairs)
            atomic_parquet(args.output_root / f"matched_pairs_{slug}_outcome_blind.parquet", pairs)
        print(
            f"      {family}: eligible={row['eligible_candidates']:,} "
            f"pairs={row['pairs_after_repair']:,} D/V={row['discovery_pairs']:,}/{row['validation_pairs']:,} "
            f"max|SMD|={safe_float(row['max_abs_smd']):.4f} PASS={bool(row['feasible_before_outcomes'])}"
        )

    gate = pd.DataFrame(gate_rows)
    all_pairs = pd.concat(pair_parts, ignore_index=True, sort=False) if pair_parts else pd.DataFrame()

    print("[4/6] Auditing discovery/validation and direction representation...")
    subgroups = subgroup_rows(d, definitions)

    print("[5/6] Writing outcome-blind manifests, gates, and invariants...")
    atomic_csv(args.report_root / "exposure_definitions.csv", definitions)
    atomic_csv(args.report_root / "family_matching_feasibility.csv", gate)
    atomic_csv(args.report_root / "subgroup_representation.csv", subgroups)
    if len(all_pairs):
        atomic_csv(args.report_root / "all_family_pair_manifest.csv", all_pairs)

    # Compact candidate manifest contains only causal variables/exposures needed to audit this phase.
    candidate_manifest_cols = [
        "candidate_id", "candidate_date", "direction", "base_mechanics_eligible",
        "mark_event_found", "basis_valid", "directional_basis_bps",
        "w30s_prior_found", "w30s_directional_basis_change_bps",
        "funding_rate", "directional_funding_rate",
        "exp_current_directional_basis", "exp_directional_basis_change_30s",
        "exp_current_directional_funding",
        *CONTROL_COLUMNS,
    ]
    assert_causal_names(candidate_manifest_cols, "7M.8B candidate manifest")
    atomic_parquet(args.output_root / "candidate_exposure_manifest_outcome_blind.parquet", d[candidate_manifest_cols].copy())

    all_pass = bool(gate["feasible_before_outcomes"].all()) if len(gate) else False
    inv = pd.DataFrame([
        {"invariant": "future_outcomes_opened", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_market_replay_performed", "pass": True, "actual": False, "required": False},
        {"invariant": "composite_signal_built", "pass": True, "actual": False, "required": False},
        {"invariant": "families_tested_independently", "pass": True, "actual": 3, "required": 3},
        {"invariant": "exact_match_cells", "pass": True, "actual": "candidate_date+direction", "required": "candidate_date+direction"},
        {"invariant": "z_caliper", "pass": True, "actual": MATCH_Z_CALIPER, "required": 0.75},
        {"invariant": "euclidean_caliper", "pass": True, "actual": MATCH_EUCLIDEAN_CALIPER, "required": 1.00},
        {"invariant": "balance_gate", "pass": True, "actual": BALANCE_SMD_GATE, "required": 0.20},
        {"invariant": "full_pair_gate", "pass": True, "actual": MIN_FULL_PAIRS, "required": 250},
        {"invariant": "split_pair_gate", "pass": True, "actual": MIN_SPLIT_PAIRS, "required": 100},
        {"invariant": "basis_semantics", "pass": True, "actual": "mark-index dislocation; no causal-source identity", "required": "conservative"},
        {"invariant": "basis_change_semantics", "pass": True, "actual": "direction-aligned 30s dislocation change before decision point", "required": "conservative"},
        {"invariant": "funding_semantics", "pass": True, "actual": "slow contract-level crowding/regime state; not short-horizon flow", "required": "conservative"},
        {"invariant": "all_three_family_gates_pass", "pass": all_pass, "actual": all_pass, "required": True},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)

    findings = make_findings(gate, definitions, subgroups)
    atomic_text(args.report_root / "findings.md", findings)
    summary = {
        "phase": PHASE,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "symbol": SYMBOL,
        "candidate_count": int(len(d)),
        "base_mechanics_eligible": int(bool_series(d["base_mechanics_eligible"]).sum()),
        "future_outcomes_opened": False,
        "raw_market_replay_performed": False,
        "composite_signal_built": False,
        "matching": {
            "exact_cells": ["candidate_date", "direction"],
            "controls": CONTROL_COLUMNS,
            "z_caliper": MATCH_Z_CALIPER,
            "euclidean_caliper": MATCH_EUCLIDEAN_CALIPER,
            "balance_smd_gate": BALANCE_SMD_GATE,
            "full_pair_gate": MIN_FULL_PAIRS,
            "split_pair_gate": MIN_SPLIT_PAIRS,
        },
        "families": gate.to_dict(orient="records"),
        "all_three_family_gates_pass": all_pass,
        "audit_source_sha256": sha256_file(audit_path),
        "next_step": (
            "Review the three outcome-blind family gates. Only passing families may proceed to a separately authorized outcome phase using their frozen pair manifests."
        ),
    }
    atomic_json(args.report_root / "summary.json", summary)

    print("[6/6] Complete.")
    print(f"      report={args.report_root}")
    print(f"      output={args.output_root}")
    print(f"      outcomes_opened=NO composite=NO all_family_pass={all_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
