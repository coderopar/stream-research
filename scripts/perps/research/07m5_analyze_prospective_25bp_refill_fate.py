#!/usr/bin/env python3
"""
Phase 7M.5 — Prospective 25-bp opposing refill-fate attribution
================================================================

PLAIN-ENGLISH QUESTION
----------------------
Phase 7M.4 established two important facts at genuinely prospective 25-bp
BTCUSDT expansion candidates:

1) the cheap early opposing-withdrawal proxy corresponds to real destruction /
   non-retention of the original opposing queue; and
2) continuation is NOT characterized by an absence of opposing refill.  More
   opposing same-price refill accompanied more continuation.

That leaves a sharper mechanistic question:

    When opposing liquidity comes back after being depleted, does it become
    durable resistance, or is it active but unable to make the resistance
    stick?

Phase 7M.5 tests that question using ONLY the already-frozen Phase 7M.3 exact
mechanics.  It does not replay depth or trades and does not regenerate
candidates.

WHY THIS DESIGN IS DIFFERENT FROM 7M.4
--------------------------------------
The primary comparison explicitly matches on the amount of initial opposing
queue destruction itself.  Therefore the main 7M.5 question is conditional:

    Among candidates with similar date, direction, short-term momentum,
    executed directional aggression, AND similar initial opposing queue
    destruction, does the subsequent refill/restoration/durability state
    distinguish future continuation?

This prevents a refill-fate variable from merely standing in for "the original
queue was damaged more."

NO composite "futile refill" score is constructed.  Each mechanism is tested
individually.

FROZEN CAUSAL SUBSTRATE
-----------------------
Phase 7M.3:
    prospective_25bp_exact_participant_interaction_v1
    builder 1.0.0
    7,067 candidates
    horizons 1 / 3 / 5 / 10 seconds
    policy EARLY_25BP_RESET5_RELEASE10_REARM60

PRIMARY OBSERVATION HORIZON
---------------------------
3 seconds.

1 / 5 / 10 seconds are robustness only.

PRIMARY OUTCOME
---------------
Origin reaches 75 bps within 15 minutes from the exact decision timestamp.

Secondary 15-minute targets are 35 / 50 / 100 bps when present in the frozen
outcome table.  P75 robustness uses 5 / 10 / 30 minutes.

OUTCOME-BLIND REFILL-FATE FEATURES
---------------------------------
Higher ORIENTED value means the predeclared "active but ineffective opposing
resistance" direction.

1) opposing__same_price_refill_to_depletion_ratio          +
   More refill relative to depletion.  This is an anchor/replication of the
   unexpected 7M.4 result; it is not counted as a new independent discovery.

2) opposing__refilled_depleted_level_share                 +
   A larger share of depleted original price levels receives refill.

3) opposing__weighted_median_first_refill_latency_ms       -
   Faster refill.  This feature can be undefined when no refill occurs; its
   analysis is therefore explicitly conditional on the feature being observed.

4) opposing__near_restoration_to_depletion_proxy           +
   More near-price restoration relative to depletion.

5) opposing__durable_same_price_refill_fraction            -
   A smaller fraction of same-price refill remains durable by the exact
   decision-time definition.

6) opposing__same_price_current_to_initial_ratio           -
   Less same-price resistance remains/restores relative to the initial state at
   the decision endpoint, AFTER matching on initial queue destruction.

DESTRUCTION ANCHOR / CONTROL
----------------------------
opposing__peak_initial_queue_loss_share

Only ONE queue-destruction representation is used as a matching control.  We do
not simultaneously control on the closely related retained/unrecovered
representations.

MATCHING
--------
Exact blocks:
    candidate_date
    direction

Primary tier: DESTRUCTION_MATCHED
    3s directional momentum
    log1p(3s directional aggressive quote)
    opposing peak initial queue-loss share

Strict tier: DESTRUCTION_STATIC
    all primary controls
    log1p(exact initial opposing frozen 5-bp quote)
    log1p(exact initial support frozen 5-bp quote)

Deterministic 1:1 matching without replacement:
    per-control |z difference| <= 0.75
    Euclidean z distance <= 1.00
    outcome-blind deterministic balance repair
    every control |SMD| <= 0.20

PRE-OUTCOME FEASIBILITY GATES
-----------------------------
FULL primary / strict comparison: >= 250 balanced pairs
DISCOVERY:                         >= 100 pairs
VALIDATION:                        >= 100 pairs
REACCEL diagnostic:                >= 75 balanced pairs

If a test fails before outcomes are loaded, it stays NOT TESTED.  Do not loosen
matching after seeing labels.

SCIENTIFIC SEQUENCE
-------------------
A. Load causal 7M.3 mechanics only.
B. Outcome-blind physical refill-fate audit.
C. Outcome-blind matching and feasibility.
D. Write/freeze analysis_inputs + matched_pairs + gate reports.
E. ONLY THEN load future outcome labels.
F. Estimate primary and robustness contrasts with candidate-day clustered
   bootstrap uncertainty for preregistered primary/split/reaccel tests.

IMPORTANT INTERPRETATION LIMITS
-------------------------------
* This is mechanistic observational research, not a randomized causal estimate.
* A positive fast/high-refill result alone does NOT prove refill is futile.
* The strongest support for "active but ineffective resistance" would be a
  coherent pattern after DESTRUCTION_MATCHED control: active/broad/fast refill
  alongside lower durability / weaker endpoint restoration, with future
  continuation separating in the preregistered directions.
* If durable restoration is instead continuation-favoring, the working
  "opposing durability failure" story must be weakened.
* No profitability or alpha claim is made.

INPUTS
------
data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/
    candidate_inputs.parquet
    candidate_mechanics.parquet
    candidate_decision_outcomes.parquet
    level_audit/candidate_date=*/part.parquet
    metadata/latest_summary.json

Optional causal cache from 7M.4, used only if present and validated:
data/derived/perps/research/prospective_25bp_exact_mechanism_analysis_v1/
    exact_initial_book_state.parquet

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_refill_fate_analysis_v1/
    analysis_inputs.parquet
    matched_pairs.parquet
    exact_initial_book_state.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_25bp_refill_fate_analysis/
    refill_fate_physical_audit.csv
    refill_fate_by_destruction_tercile.csv
    precompute_feasibility_gate.csv
    matching_balance.csv
    primary_refill_fate_contrasts.csv
    discovery_validation.csv
    static_state_robustness.csv
    observation_horizon_robustness.csv
    outcome_horizon_robustness.csv
    direction_robustness.csv
    feature_day_sign_consistency.csv
    reacceleration_diagnostic.csv
    analysis_invariants.csv
    findings.md
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


BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_refill_fate_analysis_v1"

FROZEN_PHASE7M3_BUILDER = "1.0.0"
FROZEN_PHASE7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"
FROZEN_PHASE7M4_SCHEMA = "prospective_25bp_exact_mechanism_analysis_v1"
FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"

EXPECTED_CANDIDATES = 7_067
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_OBSERVATION_HORIZON_SEC = 3
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75

DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

DESTRUCTION_ANCHOR = "opposing__peak_initial_queue_loss_share"
REFILL_INTENSITY_ANCHOR = "opposing__same_price_refill_to_depletion_ratio"

# Higher ORIENTED value = preregistered active-but-ineffective-resistance state.
REFILL_FATE_FEATURES: tuple[tuple[str, int, str, str], ...] = (
    (
        "opposing__same_price_refill_to_depletion_ratio",
        +1,
        "ACTIVE_REFILL_ANCHOR",
        "more same-price refill relative to depletion",
    ),
    (
        "opposing__refilled_depleted_level_share",
        +1,
        "REFILL_BREADTH",
        "more depleted original levels receive refill",
    ),
    (
        "opposing__weighted_median_first_refill_latency_ms",
        -1,
        "REFILL_SPEED",
        "faster first refill",
    ),
    (
        "opposing__near_restoration_to_depletion_proxy",
        +1,
        "RESTORATION_ACTIVITY",
        "more near-price restoration relative to depletion",
    ),
    (
        "opposing__durable_same_price_refill_fraction",
        -1,
        "REFILL_DURABILITY",
        "lower durable fraction of same-price refill",
    ),
    (
        "opposing__same_price_current_to_initial_ratio",
        -1,
        "ENDPOINT_RESISTANCE",
        "less same-price resistance at the endpoint relative to initial",
    ),
)

PRIMARY_MOMENTUM_CONTROL = "directional_mid_move_bps"
PRIMARY_AGGRESSION_CONTROL = "directional_aggressive_quote"

PRIMARY_CONTROL_COLUMNS = (
    "__control_momentum",
    "__control_aggression",
    "__control_destruction",
)
STRICT_CONTROL_COLUMNS = (
    "__control_momentum",
    "__control_aggression",
    "__control_destruction",
    "__control_initial_opposing",
    "__control_initial_support",
)
PRIMARY_CONTROL_TIER = "DESTRUCTION_MATCHED"
STRICT_CONTROL_TIER = "DESTRUCTION_STATIC"

FEATURE_BINS = 3
DESTRUCTION_BINS = 3
MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20

MIN_FULL_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_REACCEL_PAIRS = 75
MIN_DIRECTION_PAIRS = 100

BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260901
MIN_BOOTSTRAP_VALID_REPS = 250

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_M3_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7M3_SCHEMA
DEFAULT_M4_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7M4_SCHEMA
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_refill_fate_analysis"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.5 requires pandas, numpy and pyarrow in the project .venv.")


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def sha256_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in ids)).encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m5-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m5-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m5-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def percentile_bin(s: pd.Series, bins: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if not mask.any():
        return out
    ranks = x.loc[mask].rank(method="first", pct=True)
    out.loc[mask] = np.ceil(ranks * bins).clip(1, bins).astype(int).astype("Int64")
    return out


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mu = float(x.mean()) if x.notna().any() else math.nan
    sd = float(x.std(ddof=1)) if x.notna().sum() >= 2 else math.nan
    if not math.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (x - mu) / sd


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    den = math.sqrt(max(0.0, 0.5 * (vx + vy)))
    if den <= 1e-12:
        return 0.0 if math.isclose(float(x.mean()), float(y.mean())) else math.inf
    return float((x.mean() - y.mean()) / den)


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return math.nan
    return safe_float(x.loc[mask].rank(method="average").corr(y.loc[mask].rank(method="average")))


def robust_scale_diff(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if len(x) < 3 or len(y) < 3:
        return math.nan
    sx = (x.quantile(0.75) - x.quantile(0.25)) / 1.349
    sy = (y.quantile(0.75) - y.quantile(0.25)) / 1.349
    den = math.sqrt(0.5 * (sx * sx + sy * sy))
    if not math.isfinite(den) or den <= 1e-12:
        return math.nan
    return safe_float((x.median() - y.median()) / den)


def future_label_like_columns(columns: Iterable[str]) -> list[str]:
    """Explicit label patterns only; avoid 7M.4's false-positive 'outcome' substring rule."""
    bad = []
    for c in columns:
        lc = str(c).lower()
        if (
            lc.startswith("h300_")
            or lc.startswith("h600_")
            or lc.startswith("h900_")
            or lc.startswith("h1800_")
            or "origin_reach_" in lc
            or "fully_observed" in lc
            or "mfe_from_decision" in lc
            or "mae_from_decision" in lc
            or "seconds_to_origin" in lc
        ):
            bad.append(str(c))
    return bad


# ---------------------------------------------------------------------------
# Frozen inputs and causal/static construction
# ---------------------------------------------------------------------------

def load_metadata(root: Path) -> dict[str, Any]:
    p = root / "metadata" / "latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing Phase 7M.3 metadata: {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if str(meta.get("builder_version")) != FROZEN_PHASE7M3_BUILDER:
        raise RuntimeError(
            f"Phase 7M.3 builder mismatch: expected {FROZEN_PHASE7M3_BUILDER}, found {meta.get('builder_version')!r}"
        )
    if str(meta.get("schema_name")) != FROZEN_PHASE7M3_SCHEMA:
        raise RuntimeError(
            f"Phase 7M.3 schema mismatch: expected {FROZEN_PHASE7M3_SCHEMA}, found {meta.get('schema_name')!r}"
        )
    if str(meta.get("status")) != "PASS":
        raise RuntimeError(f"Phase 7M.3 must be PASS; found status={meta.get('status')!r}")
    if str(meta.get("run_scope")) != "FULL":
        raise RuntimeError(f"Phase 7M.3 must be FULL; found {meta.get('run_scope')!r}")
    if int(meta.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} Phase 7M.3 candidates; metadata reports {meta.get('candidate_count')!r}"
        )
    if safe_float(meta.get("candidate_threshold_bps")) != 25.0:
        raise RuntimeError("Phase 7M.3 candidate threshold must remain frozen at 25 bps.")
    if not bool(meta.get("prospective_population_includes_failed_before35")):
        raise RuntimeError("Phase 7M.3 metadata does not confirm failed-before-35 inclusion.")
    return meta


def load_frozen_inputs(m3_root: Path):
    require_stack()
    meta = load_metadata(m3_root)
    mech_path = m3_root / "candidate_mechanics.parquet"
    input_path = m3_root / "candidate_inputs.parquet"
    outcome_path = m3_root / "candidate_decision_outcomes.parquet"
    for p in (mech_path, input_path, outcome_path):
        if not p.exists():
            raise RuntimeError(f"Missing frozen Phase 7M.3 input: {p}")

    mechanics = pq.read_table(mech_path).to_pandas()
    candidates = pq.read_table(input_path).to_pandas()

    feature_names = [x[0] for x in REFILL_FATE_FEATURES]
    require_columns(
        mechanics,
        [
            "candidate_id",
            "candidate_date",
            "direction",
            "direction_label",
            "observation_horizon_sec",
            "window_valid",
            "decision_price_valid",
            PRIMARY_MOMENTUM_CONTROL,
            PRIMARY_AGGRESSION_CONTROL,
            DESTRUCTION_ANCHOR,
            *feature_names,
        ],
        "Phase 7M.3 candidate_mechanics",
    )
    require_columns(
        candidates,
        ["candidate_id", "candidate_date", "direction", "policy_id", "candidate_threshold_bps"],
        "Phase 7M.3 candidate_inputs",
    )

    candidates = candidates[candidates["policy_id"].astype(str) == FROZEN_POLICY].copy()
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("Phase 7M.3 candidate_inputs duplicate candidate IDs.")
    if len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected {EXPECTED_CANDIDATES:,} candidates; found {len(candidates):,}")

    horizons = tuple(
        sorted(
            pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce")
            .dropna().astype(int).unique().tolist()
        )
    )
    if horizons != EXPECTED_HORIZONS:
        raise RuntimeError(f"Phase 7M.3 mechanics horizons mismatch: {horizons}")
    expected_rows = EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS)
    if len(mechanics) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} mechanics rows; found {len(mechanics):,}")
    if mechanics.duplicated(["candidate_id", "observation_horizon_sec"]).any():
        raise RuntimeError("Phase 7M.3 mechanics duplicate candidate/horizon keys.")

    return meta, candidates, mechanics, outcome_path


def validate_static_cache(d: pd.DataFrame, candidate_ids: set[str]) -> bool:
    required = {
        "candidate_id",
        "initial_support_quote_5bps_exact",
        "initial_opposing_quote_5bps_exact",
        "initial_support_levels_5bps_exact",
        "initial_opposing_levels_5bps_exact",
    }
    if not required.issubset(d.columns):
        return False
    if future_label_like_columns(d.columns):
        return False
    ids = set(d["candidate_id"].astype(str))
    return ids == candidate_ids and len(d) == len(candidate_ids)


def build_exact_initial_book_state(m3_root: Path, candidate_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_stack()
    paths = sorted((m3_root / "level_audit").glob("candidate_date=*/part.parquet"))
    if not paths:
        raise RuntimeError("No frozen Phase 7M.3 level_audit date partitions found.")

    parts = []
    rows_read = 0
    key_name_seen: set[str] = set()
    for i, path in enumerate(paths, 1):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        key = "candidate_id" if "candidate_id" in names else ("event_id" if "event_id" in names else None)
        if key is None:
            raise RuntimeError(f"{path} lacks candidate_id/event_id in level audit schema.")
        required = {key, "side_role", "price", "initial_qty"}
        missing = required - names
        if missing:
            raise RuntimeError(f"{path} missing exact starting-book fields: {sorted(missing)}")

        d = pq.read_table(path, columns=[key, "side_role", "price", "initial_qty"]).to_pandas()
        rows_read += len(d)
        key_name_seen.add(key)
        d["candidate_id"] = d[key].astype(str)
        d = d[d["candidate_id"].isin(candidate_ids)].copy()
        if d.empty:
            continue
        d["side_role"] = d["side_role"].astype(str).str.upper()
        if not set(d["side_role"].dropna().unique()).issubset({"SUPPORT", "OPPOSING"}):
            raise RuntimeError(f"Unexpected exact level roles in {path}")
        price = pd.to_numeric(d["price"], errors="coerce")
        qty = pd.to_numeric(d["initial_qty"], errors="coerce")
        if ((qty <= 0) & qty.notna()).any():
            raise RuntimeError(f"Nonpositive exact initial_qty found in {path}")
        d["__initial_quote"] = price * qty
        d = d[d["__initial_quote"].notna() & np.isfinite(d["__initial_quote"]) & (d["__initial_quote"] > 0)].copy()
        g = d.groupby(["candidate_id", "side_role"])["__initial_quote"].agg(["sum", "size"]).reset_index()
        parts.append(g)
        if i == 1 or i % 10 == 0 or i == len(paths):
            print(f"      exact static book [{i:02d}/{len(paths):02d}] {path.parent.name}", flush=True)

    if not parts:
        raise RuntimeError("Exact level audit produced no candidate book state.")
    agg = pd.concat(parts, ignore_index=True)
    agg = agg.groupby(["candidate_id", "side_role"], as_index=False).agg(
        initial_quote=("sum", "sum"), initial_levels=("size", "sum")
    )
    quote = agg.pivot(index="candidate_id", columns="side_role", values="initial_quote")
    levels = agg.pivot(index="candidate_id", columns="side_role", values="initial_levels")
    out = pd.DataFrame(index=sorted(candidate_ids))
    out.index.name = "candidate_id"
    out["initial_support_quote_5bps_exact"] = quote.get("SUPPORT")
    out["initial_opposing_quote_5bps_exact"] = quote.get("OPPOSING")
    out["initial_support_levels_5bps_exact"] = levels.get("SUPPORT")
    out["initial_opposing_levels_5bps_exact"] = levels.get("OPPOSING")
    out = out.reset_index()
    both = (
        pd.to_numeric(out["initial_support_quote_5bps_exact"], errors="coerce").notna()
        & pd.to_numeric(out["initial_opposing_quote_5bps_exact"], errors="coerce").notna()
    )
    summary = {
        "source": "REBUILT_FROM_FROZEN_7M3_LEVEL_AUDIT",
        "level_partitions": len(paths),
        "level_rows_read": rows_read,
        "level_id_columns_seen": sorted(key_name_seen),
        "candidate_rows": len(out),
        "candidates_with_both_exact_sides": int(both.sum()),
        "exact_static_coverage": float(both.mean()) if len(out) else math.nan,
    }
    return out, summary


def load_or_build_static_book(m3_root: Path, m4_root: Path, candidate_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = m4_root / "exact_initial_book_state.parquet"
    if cache.exists():
        d = pq.read_table(cache).to_pandas()
        if validate_static_cache(d, candidate_ids):
            both = (
                pd.to_numeric(d["initial_support_quote_5bps_exact"], errors="coerce").notna()
                & pd.to_numeric(d["initial_opposing_quote_5bps_exact"], errors="coerce").notna()
            )
            return d, {
                "source": "REUSED_VALIDATED_7M4_CAUSAL_STATIC_CACHE",
                "cache_path": str(cache),
                "candidate_rows": len(d),
                "candidates_with_both_exact_sides": int(both.sum()),
                "exact_static_coverage": float(both.mean()) if len(d) else math.nan,
            }
        print("      7M.4 static cache exists but failed validation; rebuilding from 7M.3 level audit.", flush=True)
    return build_exact_initial_book_state(m3_root, candidate_ids)


def build_causal_inputs(candidates: pd.DataFrame, mechanics: pd.DataFrame, static_book: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = ["candidate_id", "candidate_date", "direction", "policy_id", "candidate_threshold_bps"]
    for optional in ("direction_label", "origin_kind", "is_reaccel_context"):
        if optional in candidates.columns and optional not in candidate_cols:
            candidate_cols.append(optional)
    c = candidates[candidate_cols].copy()
    c["candidate_id"] = c["candidate_id"].astype(str)
    m = mechanics.copy()
    m["candidate_id"] = m["candidate_id"].astype(str)

    merge_keys = ["candidate_id", "candidate_date", "direction"]
    if "direction_label" in c.columns and "direction_label" in m.columns:
        merge_keys.append("direction_label")
    causal = (
        m.merge(c, on=merge_keys, how="left", validate="many_to_one", suffixes=("", "__candidate"))
        .merge(static_book, on="candidate_id", how="left", validate="many_to_one")
    )

    causal["__control_momentum"] = pd.to_numeric(causal[PRIMARY_MOMENTUM_CONTROL], errors="coerce")
    causal["__control_aggression"] = np.log1p(
        pd.to_numeric(causal[PRIMARY_AGGRESSION_CONTROL], errors="coerce").clip(lower=0.0)
    )
    causal["__control_destruction"] = pd.to_numeric(causal[DESTRUCTION_ANCHOR], errors="coerce")
    causal["__control_initial_opposing"] = np.log1p(
        pd.to_numeric(causal["initial_opposing_quote_5bps_exact"], errors="coerce")
    )
    causal["__control_initial_support"] = np.log1p(
        pd.to_numeric(causal["initial_support_quote_5bps_exact"], errors="coerce")
    )

    for h in EXPECTED_HORIZONS:
        hm = pd.to_numeric(causal["observation_horizon_sec"], errors="coerce") == h
        for feature, sign, _family, _meaning in REFILL_FATE_FEATURES:
            oriented = pd.to_numeric(causal.loc[hm, feature], errors="coerce") * float(sign)
            causal.loc[hm, f"__oriented__{feature}"] = oriented
            causal.loc[hm, f"__feature_tercile__{feature}"] = percentile_bin(oriented, FEATURE_BINS).astype(float)
        destruction = pd.to_numeric(causal.loc[hm, DESTRUCTION_ANCHOR], errors="coerce")
        causal.loc[hm, "__destruction_tercile"] = percentile_bin(destruction, DESTRUCTION_BINS).astype(float)

    leaks = future_label_like_columns(causal.columns)
    if leaks:
        raise RuntimeError(f"Future labels leaked into causal analysis inputs: {leaks}")
    return causal


# ---------------------------------------------------------------------------
# Outcome-blind physical audit
# ---------------------------------------------------------------------------

def refill_fate_physical_audit(causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = causal[
        (pd.to_numeric(causal["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
        & bool_series(causal["window_valid"])
        & bool_series(causal["decision_price_valid"])
    ].copy()

    rows = []
    tercile_rows = []
    destruction = pd.to_numeric(d[DESTRUCTION_ANCHOR], errors="coerce")
    refill_anchor = pd.to_numeric(d[REFILL_INTENSITY_ANCHOR], errors="coerce")
    destruction_tercile = percentile_bin(destruction, DESTRUCTION_BINS)

    for feature, sign, family, meaning in REFILL_FATE_FEATURES:
        raw = pd.to_numeric(d[feature], errors="coerce")
        oriented = raw * float(sign)
        observed = raw.notna() & np.isfinite(raw)
        x = d.loc[observed].copy()
        xo = oriented.loc[observed]

        hi = xo.loc[pd.to_numeric(destruction_tercile.loc[observed], errors="coerce") == DESTRUCTION_BINS]
        lo = xo.loc[pd.to_numeric(destruction_tercile.loc[observed], errors="coerce") == 1]

        rows.append({
            "feature": feature,
            "feature_family": family,
            "orientation_sign": sign,
            "oriented_meaning": meaning,
            "primary_horizon_rows": int(len(d)),
            "observed_rows": int(observed.sum()),
            "observed_fraction": float(observed.mean()) if len(d) else math.nan,
            "raw_mean": safe_float(raw.loc[observed].mean()),
            "raw_median": safe_float(raw.loc[observed].median()),
            "destruction_to_oriented_spearman": spearman_rank_corr(destruction.loc[observed], xo),
            "refill_intensity_to_oriented_spearman": spearman_rank_corr(refill_anchor.loc[observed], xo),
            "high_destruction_oriented_mean": safe_float(hi.mean()),
            "low_destruction_oriented_mean": safe_float(lo.mean()),
            "high_minus_low_destruction_oriented_mean": safe_float(hi.mean() - lo.mean()),
            "robust_standardized_high_minus_low_destruction": robust_scale_diff(hi, lo),
            "analysis_condition": "FEATURE_OBSERVED_ONLY" if observed.mean() < 0.999999 else "FULL_VALID_PRIMARY_ROWS",
        })

        for tercile in (1, 2, 3):
            mask = observed & (pd.to_numeric(destruction_tercile, errors="coerce") == tercile)
            vals = oriented.loc[mask]
            tercile_rows.append({
                "feature": feature,
                "feature_family": family,
                "destruction_tercile": tercile,
                "rows": int(mask.sum()),
                "destruction_median": safe_float(destruction.loc[mask].median()),
                "oriented_feature_mean": safe_float(vals.mean()),
                "oriented_feature_median": safe_float(vals.median()),
                "raw_feature_mean": safe_float(raw.loc[mask].mean()),
                "raw_feature_median": safe_float(raw.loc[mask].median()),
            })
    return pd.DataFrame(rows), pd.DataFrame(tercile_rows)


# ---------------------------------------------------------------------------
# Outcome-blind matching
# ---------------------------------------------------------------------------

def add_z_controls(d: pd.DataFrame, controls: Sequence[str]) -> pd.DataFrame:
    x = d.copy()
    for c in controls:
        x[f"__z__{c}"] = zscore(x[c])
    return x


def greedy_match(
    high: pd.DataFrame,
    low: pd.DataFrame,
    *,
    feature: str,
    control_tier: str,
    controls: Sequence[str],
    population: str,
    observation_horizon_sec: int,
) -> pd.DataFrame:
    rows = []
    for (candidate_date, direction), hb in high.groupby(["candidate_date", "direction"], sort=True):
        lb = low[
            (low["candidate_date"].astype(str) == str(candidate_date))
            & (pd.to_numeric(low["direction"], errors="coerce") == int(direction))
        ].copy()
        if hb.empty or lb.empty:
            continue
        hb = hb.sort_values([f"__oriented__{feature}", "candidate_id"], ascending=[False, True], kind="mergesort")
        lb = lb.sort_values("candidate_id", kind="mergesort")
        available = set(lb.index.tolist())
        for _, hr in hb.iterrows():
            best_idx = None
            best_key = None
            best_diffs = None
            for lo_idx in sorted(available):
                lr = lb.loc[lo_idx]
                diffs = []
                valid = True
                for c in controls:
                    hv = safe_float(hr.get(f"__z__{c}"))
                    lv = safe_float(lr.get(f"__z__{c}"))
                    if not math.isfinite(hv) or not math.isfinite(lv):
                        valid = False
                        break
                    diff = hv - lv
                    if abs(diff) > MATCH_Z_CALIPER + 1e-12:
                        valid = False
                        break
                    diffs.append(diff)
                if not valid:
                    continue
                dist = math.sqrt(sum(v * v for v in diffs))
                if dist > MATCH_EUCLIDEAN_CALIPER + 1e-12:
                    continue
                key = (dist, str(lr["candidate_id"]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = lo_idx
                    best_diffs = diffs
            if best_idx is None:
                continue
            lr = lb.loc[best_idx]
            available.remove(best_idx)
            row = {
                "feature": feature,
                "control_tier": control_tier,
                "population": population,
                "observation_horizon_sec": int(observation_horizon_sec),
                "candidate_date": str(candidate_date),
                "direction": int(direction),
                "high_candidate_id": str(hr["candidate_id"]),
                "low_candidate_id": str(lr["candidate_id"]),
                "match_distance": float(best_key[0]),
                "high_oriented_feature": safe_float(hr[f"__oriented__{feature}"]),
                "low_oriented_feature": safe_float(lr[f"__oriented__{feature}"]),
            }
            for c, diff in zip(controls, best_diffs or []):
                row[f"high__{c}"] = safe_float(hr[c])
                row[f"low__{c}"] = safe_float(lr[c])
                row[f"high__z__{c}"] = safe_float(hr[f"__z__{c}"])
                row[f"low__z__{c}"] = safe_float(lr[f"__z__{c}"])
                row[f"pair_z_diff__{c}"] = safe_float(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def pair_balance(pairs: pd.DataFrame, controls: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"pairs": int(len(pairs)), "max_abs_smd": math.nan}
    vals = []
    for c in controls:
        v = smd(pairs.get(f"high__{c}", pd.Series(dtype=float)), pairs.get(f"low__{c}", pd.Series(dtype=float)))
        out[f"smd__{c}"] = v
        if math.isfinite(v):
            vals.append(abs(v))
    out["max_abs_smd"] = max(vals) if vals else math.nan
    return out


def repair_balance(pairs: pd.DataFrame, *, controls: Sequence[str], min_pairs: int) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    x = pairs.copy().reset_index(drop=True)
    while len(x) > min_pairs:
        b = pair_balance(x, controls)
        smds = {c: safe_float(b.get(f"smd__{c}")) for c in controls}
        finite = {c: v for c, v in smds.items() if math.isfinite(v)}
        if finite and max(abs(v) for v in finite.values()) <= BALANCE_SMD_GATE:
            break
        if not finite:
            break
        worst = max(finite, key=lambda c: (abs(finite[c]), c))
        sign = 1.0 if finite[worst] >= 0 else -1.0
        contribution = (
            pd.to_numeric(x[f"high__{worst}"], errors="coerce")
            - pd.to_numeric(x[f"low__{worst}"], errors="coerce")
        ) * sign
        order = pd.DataFrame({
            "idx": x.index,
            "contribution": contribution.fillna(-np.inf),
            "tie": x["high_candidate_id"].astype(str) + "|" + x["low_candidate_id"].astype(str),
        }).sort_values(["contribution", "tie"], ascending=[False, False], kind="mergesort")
        x = x.drop(index=int(order.iloc[0]["idx"])).reset_index(drop=True)
    return x


def population_mask(d: pd.DataFrame, population: str) -> pd.Series:
    if population == "ALL":
        return pd.Series(True, index=d.index)
    if population == "REACCEL_CONTEXT":
        if "is_reaccel_context" not in d.columns:
            return pd.Series(False, index=d.index)
        return bool_series(d["is_reaccel_context"])
    raise ValueError(population)


def build_pair_sets(causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_parts = []
    gate_rows = []
    pair_counter = 0
    configs = []

    # Primary destruction-matched tier at every observation horizon.
    for h in EXPECTED_HORIZONS:
        for feature, _sign, _family, _meaning in REFILL_FATE_FEATURES:
            configs.append((h, feature, PRIMARY_CONTROL_TIER, "ALL", PRIMARY_CONTROL_COLUMNS, MIN_FULL_PAIRS))

    # Exact starting-book robustness only at the primary horizon.
    for feature, _sign, _family, _meaning in REFILL_FATE_FEATURES:
        configs.append((PRIMARY_OBSERVATION_HORIZON_SEC, feature, STRICT_CONTROL_TIER, "ALL", STRICT_CONTROL_COLUMNS, MIN_FULL_PAIRS))

    # Reacceleration diagnostic, still destruction-matched.
    if "is_reaccel_context" in causal.columns:
        for feature, _sign, _family, _meaning in REFILL_FATE_FEATURES:
            configs.append((PRIMARY_OBSERVATION_HORIZON_SEC, feature, PRIMARY_CONTROL_TIER, "REACCEL_CONTEXT", PRIMARY_CONTROL_COLUMNS, MIN_REACCEL_PAIRS))

    for i, (h, feature, tier, population, controls, min_pairs) in enumerate(configs, 1):
        d = causal[
            (pd.to_numeric(causal["observation_horizon_sec"], errors="coerce") == int(h))
            & population_mask(causal, population)
            & bool_series(causal["window_valid"])
            & bool_series(causal["decision_price_valid"])
        ].copy()
        bcol = f"__feature_tercile__{feature}"
        need = [bcol, f"__oriented__{feature}", *controls]
        for c in need:
            d = d[pd.to_numeric(d[c], errors="coerce").notna()].copy()
        d = add_z_controls(d, controls)
        for c in controls:
            d = d[pd.to_numeric(d[f"__z__{c}"], errors="coerce").notna()].copy()
        high = d[pd.to_numeric(d[bcol], errors="coerce") == FEATURE_BINS].copy()
        low = d[pd.to_numeric(d[bcol], errors="coerce") == 1].copy()

        raw = greedy_match(
            high, low, feature=feature, control_tier=tier, controls=controls,
            population=population, observation_horizon_sec=int(h)
        )
        pre = pair_balance(raw, controls)
        repaired = repair_balance(raw, controls=controls, min_pairs=int(min_pairs))
        post = pair_balance(repaired, controls)
        balance_pass = bool(
            len(repaired) > 0
            and math.isfinite(safe_float(post["max_abs_smd"]))
            and safe_float(post["max_abs_smd"]) <= BALANCE_SMD_GATE + 1e-12
        )
        disc_n = int((repaired["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(repaired) else 0
        val_n = int((repaired["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(repaired) else 0
        total_gate = len(repaired) >= int(min_pairs) and balance_pass
        split_gate = (
            disc_n >= MIN_SPLIT_PAIRS and val_n >= MIN_SPLIT_PAIRS
            if population == "ALL" and int(h) == PRIMARY_OBSERVATION_HORIZON_SEC
            else True
        )
        feasible = bool(total_gate and split_gate)

        if len(repaired):
            repaired = repaired.copy()
            repaired["pair_id"] = [f"7M5-{pair_counter + j + 1:09d}" for j in range(len(repaired))]
            pair_counter += len(repaired)
            repaired["min_pairs_gate"] = int(min_pairs)
            repaired["final_balance_pass"] = balance_pass
            repaired["feasible_before_label_load"] = feasible
            repaired["discovery_pairs"] = disc_n
            repaired["validation_pairs"] = val_n
            pair_parts.append(repaired)

        row = {
            "observation_horizon_sec": int(h),
            "feature": feature,
            "control_tier": tier,
            "population": population,
            "eligible_rows": int(len(d)),
            "eligible_fraction_of_7067": float(len(d) / EXPECTED_CANDIDATES),
            "high_tercile_rows": int(len(high)),
            "low_tercile_rows": int(len(low)),
            "pairs_before_repair": int(len(raw)),
            "pairs_after_repair": int(len(repaired)),
            "pairs_trimmed": int(len(raw) - len(repaired)),
            "min_pairs_gate": int(min_pairs),
            "discovery_pairs": disc_n,
            "validation_pairs": val_n,
            "min_split_pairs": MIN_SPLIT_PAIRS if population == "ALL" and int(h) == PRIMARY_OBSERVATION_HORIZON_SEC else 0,
            "pre_max_abs_smd": safe_float(pre["max_abs_smd"]),
            "post_max_abs_smd": safe_float(post["max_abs_smd"]),
            "balance_smd_gate": BALANCE_SMD_GATE,
            "balance_pass": balance_pass,
            "split_gate_pass": split_gate,
            "feasible_before_label_load": feasible,
        }
        for c in controls:
            row[f"pre_smd__{c}"] = safe_float(pre.get(f"smd__{c}"))
            row[f"post_smd__{c}"] = safe_float(post.get(f"smd__{c}"))
        gate_rows.append(row)

        if i == 1 or i % 6 == 0 or i == len(configs):
            print(
                f"      matching [{i:02d}/{len(configs):02d}] h={h}s tier={tier:19s} "
                f"pop={population:16s} feature={feature[:40]} pairs={len(repaired):,} "
                f"max|SMD|={safe_float(post['max_abs_smd']):.3f} feasible={feasible}",
                flush=True,
            )

    pairs = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    gate = pd.DataFrame(gate_rows)
    return pairs, gate


# ---------------------------------------------------------------------------
# Outcomes — intentionally called only after causal manifests are written
# ---------------------------------------------------------------------------

def load_outcomes_after_freeze(path: Path) -> pd.DataFrame:
    require_stack()
    outcomes = pq.read_table(path).to_pandas()
    require_columns(
        outcomes,
        [
            "candidate_id", "observation_horizon_sec", "decision_price_valid",
            "h300_fully_observed", "h600_fully_observed", "h900_fully_observed", "h1800_fully_observed",
            "h900_origin_reach_50", "h900_origin_reach_75", "h900_origin_reach_100",
        ],
        "Phase 7M.3 candidate_decision_outcomes",
    )
    if outcomes.duplicated(["candidate_id", "observation_horizon_sec"]).any():
        raise RuntimeError("Outcome table duplicate candidate/horizon keys.")
    return outcomes


def pair_outcome_frame(pairs: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    keep = ["candidate_id", "observation_horizon_sec", "decision_price_valid"]
    for h in OUTCOME_HORIZONS_SEC:
        for c in (
            f"h{h}_fully_observed",
            f"h{h}_origin_reach_35",
            f"h{h}_origin_reach_50",
            f"h{h}_origin_reach_75",
            f"h{h}_origin_reach_100",
        ):
            if c in outcomes.columns:
                keep.append(c)
    keep = list(dict.fromkeys(keep))
    hi = outcomes[keep].add_prefix("high__")
    lo = outcomes[keep].add_prefix("low__")
    return (
        pairs.merge(
            hi,
            left_on=["high_candidate_id", "observation_horizon_sec"],
            right_on=["high__candidate_id", "high__observation_horizon_sec"],
            how="left", validate="many_to_one",
        )
        .merge(
            lo,
            left_on=["low_candidate_id", "observation_horizon_sec"],
            right_on=["low__candidate_id", "low__observation_horizon_sec"],
            how="left", validate="many_to_one",
        )
    )


def available_targets_from_joined(pairs: pd.DataFrame, h: int) -> list[int]:
    out = []
    for target in (35, 50, 75, 100):
        if f"high__h{h}_origin_reach_{target}" in pairs.columns and f"low__h{h}_origin_reach_{target}" in pairs.columns:
            out.append(target)
    return out


def paired_effect(pairs: pd.DataFrame, *, target: int, outcome_horizon_sec: int) -> dict[str, Any]:
    empty = {"pairs": 0, "high_rate": math.nan, "low_rate": math.nan, "effect_pp": math.nan, "high_wins": 0, "low_wins": 0}
    if pairs.empty:
        return empty
    fully_h = f"high__h{outcome_horizon_sec}_fully_observed"
    fully_l = f"low__h{outcome_horizon_sec}_fully_observed"
    y_h = f"high__h{outcome_horizon_sec}_origin_reach_{target}"
    y_l = f"low__h{outcome_horizon_sec}_origin_reach_{target}"
    if any(c not in pairs.columns for c in (fully_h, fully_l, y_h, y_l)):
        return empty
    obs = (
        bool_series(pairs[fully_h]) & bool_series(pairs[fully_l])
        & bool_series(pairs["high__decision_price_valid"]) & bool_series(pairs["low__decision_price_valid"])
    )
    x = pairs.loc[obs].copy()
    hy = x[y_h].astype("boolean")
    ly = x[y_l].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid]
    hy = hy.loc[valid].astype(bool)
    ly = ly.loc[valid].astype(bool)
    if not len(x):
        return empty
    diff = hy.astype(int) - ly.astype(int)
    return {
        "pairs": int(len(x)),
        "high_rate": float(hy.mean()),
        "low_rate": float(ly.mean()),
        "effect_pp": 100.0 * float(diff.mean()),
        "high_wins": int((hy & ~ly).sum()),
        "low_wins": int((~hy & ly).sum()),
    }


def bootstrap_matched_day(
    pairs: pd.DataFrame, *, target: int, outcome_horizon_sec: int, reps: int, seed: int
) -> tuple[float, float, int]:
    fully_h = f"high__h{outcome_horizon_sec}_fully_observed"
    fully_l = f"low__h{outcome_horizon_sec}_fully_observed"
    y_h = f"high__h{outcome_horizon_sec}_origin_reach_{target}"
    y_l = f"low__h{outcome_horizon_sec}_origin_reach_{target}"
    if any(c not in pairs.columns for c in (fully_h, fully_l, y_h, y_l)):
        return math.nan, math.nan, 0
    obs = (
        bool_series(pairs[fully_h]) & bool_series(pairs[fully_l])
        & bool_series(pairs["high__decision_price_valid"]) & bool_series(pairs["low__decision_price_valid"])
    )
    x = pairs.loc[obs].copy()
    hy = x[y_h].astype("boolean")
    ly = x[y_l].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid].copy()
    if x.empty:
        return math.nan, math.nan, 0
    x["__diff"] = x[y_h].astype(bool).astype(int) - x[y_l].astype(bool).astype(int)
    g = x.groupby(x["candidate_date"].astype(str))["__diff"].agg(["sum", "count"])
    if len(g) < 2:
        return math.nan, math.nan, 0
    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    n_days = len(g)
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=int(reps))
    den = multiplicity @ counts
    vals = np.divide(multiplicity @ sums, den, out=np.full(int(reps), np.nan), where=den > 0) * 100.0
    vals = vals[np.isfinite(vals)]
    if len(vals) < MIN_BOOTSTRAP_VALID_REPS:
        return math.nan, math.nan, int(len(vals))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def result_label(effect_pp: float, ci_low: float, feasible: bool) -> str:
    if not feasible or not math.isfinite(effect_pp):
        return "NOT_TESTED"
    if effect_pp <= 0:
        return "NOT_SUPPORTED"
    if math.isfinite(ci_low) and ci_low > 0:
        return "SUPPORTED"
    return "TENTATIVE"


def gate_map(gate: pd.DataFrame) -> dict[tuple[int, str, str, str], bool]:
    return {
        (int(r.observation_horizon_sec), str(r.feature), str(r.control_tier), str(r.population)): bool(r.feasible_before_label_load)
        for r in gate.itertuples(index=False)
    }


def feature_contract(feature: str) -> tuple[int, str, str]:
    for f, sign, family, meaning in REFILL_FATE_FEATURES:
        if f == feature:
            return sign, family, meaning
    raise KeyError(feature)


# ---------------------------------------------------------------------------
# Post-outcome analyses
# ---------------------------------------------------------------------------

def analyze_primary(joined: pd.DataFrame, gate: pd.DataFrame, bootstrap_reps: int):
    gm = gate_map(gate)
    primary_rows, split_rows, static_rows = [], [], []
    for feature, sign, family, meaning in REFILL_FATE_FEATURES:
        for tier in (PRIMARY_CONTROL_TIER, STRICT_CONTROL_TIER):
            p = joined[
                (pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
                & (joined["feature"] == feature)
                & (joined["control_tier"] == tier)
                & (joined["population"] == "ALL")
            ].copy()
            feasible = gm.get((PRIMARY_OBSERVATION_HORIZON_SEC, feature, tier, "ALL"), False)
            for target in available_targets_from_joined(p, PRIMARY_OUTCOME_HORIZON_SEC):
                r = paired_effect(p, target=target, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC)
                if feasible:
                    lo, hi, nboot = bootstrap_matched_day(
                        p, target=target, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                        reps=bootstrap_reps,
                        seed=BOOTSTRAP_SEED + sum(ord(c) for c in feature + tier) + target * 100,
                    )
                else:
                    lo = hi = math.nan; nboot = 0
                row = {
                    "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                    "feature_orientation_sign": sign, "control_tier": tier,
                    "target_bps": int(target), "observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
                    "outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
                    "feasible_before_label_load": feasible, **r,
                    "ci_low_pp": lo, "ci_high_pp": hi, "bootstrap_valid_reps": nboot,
                    "interpretation": result_label(safe_float(r["effect_pp"]), lo, feasible),
                }
                if tier == PRIMARY_CONTROL_TIER:
                    primary_rows.append(row)
                else:
                    static_rows.append(row)

            # Discovery/validation: P75 only, both tiers.
            for split_name, mask in (
                ("DISCOVERY", p["candidate_date"].astype(str) <= DISCOVERY_END_DATE),
                ("VALIDATION", p["candidate_date"].astype(str) >= VALIDATION_START_DATE),
            ):
                sp = p.loc[mask].copy()
                split_feasible = bool(feasible and len(sp) >= MIN_SPLIT_PAIRS)
                r = paired_effect(sp, target=PRIMARY_TARGET_BPS, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC)
                if split_feasible:
                    lo, hi, nboot = bootstrap_matched_day(
                        sp, target=PRIMARY_TARGET_BPS, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                        reps=bootstrap_reps,
                        seed=BOOTSTRAP_SEED + 200_000 + sum(ord(c) for c in feature + tier + split_name),
                    )
                else:
                    lo = hi = math.nan; nboot = 0
                split_rows.append({
                    "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                    "control_tier": tier, "split": split_name,
                    "pairs_available_before_outcome_filter": int(len(sp)),
                    "split_feasible_before_label_load": split_feasible, **r,
                    "ci_low_pp": lo, "ci_high_pp": hi, "bootstrap_valid_reps": nboot,
                    "interpretation": result_label(safe_float(r["effect_pp"]), lo, split_feasible),
                })
    return pd.DataFrame(primary_rows), pd.DataFrame(split_rows), pd.DataFrame(static_rows)


def analyze_observation_horizons(joined: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    gm = gate_map(gate)
    rows = []
    for feature, sign, family, meaning in REFILL_FATE_FEATURES:
        for h in EXPECTED_HORIZONS:
            p = joined[
                (pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == h)
                & (joined["feature"] == feature)
                & (joined["control_tier"] == PRIMARY_CONTROL_TIER)
                & (joined["population"] == "ALL")
            ].copy()
            feasible = gm.get((h, feature, PRIMARY_CONTROL_TIER, "ALL"), False)
            r = paired_effect(p, target=PRIMARY_TARGET_BPS, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC)
            rows.append({
                "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                "feature_orientation_sign": sign, "control_tier": PRIMARY_CONTROL_TIER,
                "observation_horizon_sec": h, "target_bps": PRIMARY_TARGET_BPS,
                "outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
                "feasible_before_label_load": feasible, **r,
                "role": "PRIMARY" if h == PRIMARY_OBSERVATION_HORIZON_SEC else "ROBUSTNESS_ONLY",
            })
    return pd.DataFrame(rows)


def analyze_outcome_horizons(joined: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    gm = gate_map(gate)
    rows = []
    for feature, sign, family, meaning in REFILL_FATE_FEATURES:
        for tier in (PRIMARY_CONTROL_TIER, STRICT_CONTROL_TIER):
            p = joined[
                (pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
                & (joined["feature"] == feature)
                & (joined["control_tier"] == tier)
                & (joined["population"] == "ALL")
            ].copy()
            feasible = gm.get((PRIMARY_OBSERVATION_HORIZON_SEC, feature, tier, "ALL"), False)
            for oh in OUTCOME_HORIZONS_SEC:
                r = paired_effect(p, target=PRIMARY_TARGET_BPS, outcome_horizon_sec=oh)
                rows.append({
                    "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                    "feature_orientation_sign": sign, "control_tier": tier,
                    "observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
                    "outcome_horizon_sec": oh, "target_bps": PRIMARY_TARGET_BPS,
                    "feasible_before_label_load": feasible, **r,
                    "role": "PRIMARY" if oh == PRIMARY_OUTCOME_HORIZON_SEC else "ROBUSTNESS_ONLY",
                })
    return pd.DataFrame(rows)


def analyze_directions(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, _sign, family, meaning in REFILL_FATE_FEATURES:
        for tier in (PRIMARY_CONTROL_TIER, STRICT_CONTROL_TIER):
            p = joined[
                (pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
                & (joined["feature"] == feature)
                & (joined["control_tier"] == tier)
                & (joined["population"] == "ALL")
            ].copy()
            for direction, label in ((1, "UP"), (-1, "DOWN")):
                sp = p[pd.to_numeric(p["direction"], errors="coerce") == direction].copy()
                r = paired_effect(sp, target=PRIMARY_TARGET_BPS, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC)
                rows.append({
                    "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                    "control_tier": tier, "direction": direction, "direction_label": label,
                    "pairs_available_before_outcome_filter": int(len(sp)),
                    "direction_pair_gate": MIN_DIRECTION_PAIRS,
                    "direction_diagnostic_feasible": bool(len(sp) >= MIN_DIRECTION_PAIRS), **r,
                })
    return pd.DataFrame(rows)


def analyze_reaccel(joined: pd.DataFrame, gate: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    if joined.empty or not (joined["population"] == "REACCEL_CONTEXT").any():
        return pd.DataFrame()
    gm = gate_map(gate)
    rows = []
    for feature, sign, family, meaning in REFILL_FATE_FEATURES:
        p = joined[
            (pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
            & (joined["feature"] == feature)
            & (joined["control_tier"] == PRIMARY_CONTROL_TIER)
            & (joined["population"] == "REACCEL_CONTEXT")
        ].copy()
        feasible = gm.get((PRIMARY_OBSERVATION_HORIZON_SEC, feature, PRIMARY_CONTROL_TIER, "REACCEL_CONTEXT"), False)
        for target in available_targets_from_joined(p, PRIMARY_OUTCOME_HORIZON_SEC):
            r = paired_effect(p, target=target, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC)
            if feasible:
                lo, hi, nboot = bootstrap_matched_day(
                    p, target=target, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                    reps=bootstrap_reps,
                    seed=BOOTSTRAP_SEED + 700_000 + sum(ord(c) for c in feature) + target * 10,
                )
            else:
                lo = hi = math.nan; nboot = 0
            rows.append({
                "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                "feature_orientation_sign": sign, "target_bps": target,
                "feasible_before_label_load": feasible, **r,
                "ci_low_pp": lo, "ci_high_pp": hi, "bootstrap_valid_reps": nboot,
                "interpretation": result_label(safe_float(r["effect_pp"]), lo, feasible),
            })
    return pd.DataFrame(rows)


def day_sign_consistency(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, _sign, family, meaning in REFILL_FATE_FEATURES:
        for tier in (PRIMARY_CONTROL_TIER, STRICT_CONTROL_TIER):
            p = joined[
                (pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
                & (joined["feature"] == feature)
                & (joined["control_tier"] == tier)
                & (joined["population"] == "ALL")
            ].copy()
            effects = []
            for day, g in p.groupby(p["candidate_date"].astype(str)):
                r = paired_effect(g, target=PRIMARY_TARGET_BPS, outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC)
                if int(r["pairs"]) >= 3 and math.isfinite(safe_float(r["effect_pp"])):
                    effects.append((str(day), safe_float(r["effect_pp"])))
            vals = [v for _, v in effects]
            nonzero = [v for v in vals if v != 0]
            rows.append({
                "feature": feature, "feature_family": family, "oriented_meaning": meaning,
                "control_tier": tier, "comparable_days": len(vals),
                "positive_days": sum(v > 0 for v in vals), "negative_days": sum(v < 0 for v in vals),
                "zero_days": sum(v == 0 for v in vals),
                "positive_share_among_nonzero_days": (sum(v > 0 for v in nonzero) / len(nonzero)) if nonzero else math.nan,
                "median_day_effect_pp": float(np.median(vals)) if vals else math.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Invariants and findings
# ---------------------------------------------------------------------------

def build_invariants(
    *, causal: pd.DataFrame, pairs: pd.DataFrame, gate: pd.DataFrame,
    outcomes: pd.DataFrame, static_book: pd.DataFrame, m3_meta: dict[str, Any],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({"invariant": name, "status": "PASS" if bool(ok) else "FAIL", "observed": observed, "expected": expected})

    candidate_n = causal["candidate_id"].astype(str).nunique()
    add("expected_candidate_population", candidate_n == EXPECTED_CANDIDATES, candidate_n, EXPECTED_CANDIDATES)
    horizons = tuple(sorted(pd.to_numeric(causal["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist()))
    add("expected_horizons", horizons == EXPECTED_HORIZONS, horizons, EXPECTED_HORIZONS)
    add("causal_row_count", len(causal) == EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS), len(causal), EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS))

    causal_leaks = future_label_like_columns(causal.columns)
    pair_leaks = future_label_like_columns(pairs.columns)
    add("causal_analysis_inputs_future_clean", len(causal_leaks) == 0, causal_leaks, [])
    add("matched_pair_manifest_future_clean", len(pair_leaks) == 0, pair_leaks, [])

    feasible_bad = gate[
        gate["feasible_before_label_load"].astype(bool)
        & (pd.to_numeric(gate["post_max_abs_smd"], errors="coerce") > BALANCE_SMD_GATE + 1e-12)
    ]
    add("all_feasible_matches_balanced", len(feasible_bad) == 0, len(feasible_bad), 0)

    static_cov = float((
        pd.to_numeric(static_book["initial_support_quote_5bps_exact"], errors="coerce").notna()
        & pd.to_numeric(static_book["initial_opposing_quote_5bps_exact"], errors="coerce").notna()
    ).mean())
    add("exact_static_book_coverage_at_least_95pct", static_cov >= 0.95, static_cov, ">=0.95")

    p3 = causal[pd.to_numeric(causal["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC]
    mech_cov = float(bool_series(p3["window_valid"]).mean())
    add("primary_3s_exact_mechanics_coverage_at_least_95pct", mech_cov >= 0.95, mech_cov, ">=0.95")

    add("phase7m3_metadata_candidate_count", int(m3_meta.get("candidate_count", -1)) == EXPECTED_CANDIDATES, m3_meta.get("candidate_count"), EXPECTED_CANDIDATES)
    add("outcome_keys_unique", not outcomes.duplicated(["candidate_id", "observation_horizon_sec"]).any(), int(outcomes.duplicated(["candidate_id", "observation_horizon_sec"]).sum()), 0)
    add("physical_audit_has_all_predeclared_features", len(audit) == len(REFILL_FATE_FEATURES), len(audit), len(REFILL_FATE_FEATURES))

    primary_gate = gate[
        (pd.to_numeric(gate["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
        & (gate["control_tier"] == PRIMARY_CONTROL_TIER)
        & (gate["population"] == "ALL")
    ]
    strict_gate = gate[
        (pd.to_numeric(gate["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
        & (gate["control_tier"] == STRICT_CONTROL_TIER)
        & (gate["population"] == "ALL")
    ]
    add("all_primary_features_have_gate_rows", len(primary_gate) == len(REFILL_FATE_FEATURES), len(primary_gate), len(REFILL_FATE_FEATURES))
    add("all_strict_static_features_have_gate_rows", len(strict_gate) == len(REFILL_FATE_FEATURES), len(strict_gate), len(REFILL_FATE_FEATURES))

    # Verify destruction is genuinely present in primary matching controls.
    destruction_smd_col = f"post_smd____control_destruction"
    add("primary_matching_includes_destruction_control", destruction_smd_col in gate.columns, destruction_smd_col in gate.columns, True)

    return pd.DataFrame(rows)


def make_findings(
    *, audit: pd.DataFrame, gate: pd.DataFrame, primary: pd.DataFrame,
    split: pd.DataFrame, static: pd.DataFrame, obs_robust: pd.DataFrame,
    outcome_robust: pd.DataFrame, reaccel: pd.DataFrame, invariants: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7M.5 — Prospective 25-bp opposing refill-fate attribution",
        "",
        "This file is generated mechanically from the frozen analysis outputs.  Interpret with the preregistered design; do not rescue failed matches after seeing outcomes.",
        "",
        f"- Primary observation horizon: **{PRIMARY_OBSERVATION_HORIZON_SEC}s**",
        f"- Primary outcome: **P{PRIMARY_TARGET_BPS} within {PRIMARY_OUTCOME_HORIZON_SEC//60}m**",
        f"- Primary matching tier: **{PRIMARY_CONTROL_TIER}** = momentum + aggression + peak opposing initial-queue destruction",
        f"- Strict tier: **{STRICT_CONTROL_TIER}** additionally controls exact initial support/opposing 5-bp quote",
        "- No raw replay, no ML, no composite refill score.",
        "",
        "## Outcome-blind physical audit",
        "",
    ]
    for _, r in audit.iterrows():
        lines.append(
            f"- `{r['feature']}`: observed {100*safe_float(r['observed_fraction']):.1f}% of valid 3s rows; "
            f"destruction→oriented rho={safe_float(r['destruction_to_oriented_spearman']):+.3f}; "
            f"refill-intensity→oriented rho={safe_float(r['refill_intensity_to_oriented_spearman']):+.3f}."
        )

    lines += ["", "## Pre-outcome feasibility at 3s", ""]
    pg = gate[
        (pd.to_numeric(gate["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
        & (gate["population"] == "ALL")
    ]
    for _, r in pg.iterrows():
        lines.append(
            f"- `{r['feature']}` / {r['control_tier']}: pairs={int(r['pairs_after_repair'])}, "
            f"D/V={int(r['discovery_pairs'])}/{int(r['validation_pairs'])}, max|SMD|={safe_float(r['post_max_abs_smd']):.3f}, "
            f"feasible={bool(r['feasible_before_label_load'])}."
        )

    lines += ["", "## Primary P75/15m destruction-matched contrasts", ""]
    p75 = primary[primary["target_bps"] == PRIMARY_TARGET_BPS] if not primary.empty else primary
    for _, r in p75.iterrows():
        lines.append(
            f"- `{r['feature']}`: high={100*safe_float(r['high_rate']):.2f}%, low={100*safe_float(r['low_rate']):.2f}%, "
            f"effect={safe_float(r['effect_pp']):+.2f} pp, CI [{safe_float(r['ci_low_pp']):+.2f}, {safe_float(r['ci_high_pp']):+.2f}], "
            f"{r['interpretation']}."
        )

    lines += ["", "## Strict static P75/15m robustness", ""]
    s75 = static[static["target_bps"] == PRIMARY_TARGET_BPS] if not static.empty else static
    for _, r in s75.iterrows():
        lines.append(
            f"- `{r['feature']}`: feasible={bool(r['feasible_before_label_load'])}, effect={safe_float(r['effect_pp']):+.2f} pp, "
            f"CI [{safe_float(r['ci_low_pp']):+.2f}, {safe_float(r['ci_high_pp']):+.2f}], {r['interpretation']}."
        )

    lines += ["", "## Invariants", ""]
    for _, r in invariants.iterrows():
        lines.append(f"- **{r['status']}** `{r['invariant']}` — observed={r['observed']} expected={r['expected']}")

    lines += [
        "",
        "## Interpretation rule",
        "",
        "Do not call the active-but-ineffective-resistance mechanism established from refill amount alone.  The strongest support requires coherent destruction-matched evidence across the separately tested activity/breadth/speed/restoration/durability/end-state variables.  Failed feasibility rows remain NOT TESTED.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    # Future-label detector must NOT repeat the 7M.4 false positive.
    assert future_label_like_columns(["feasible_before_outcomes", "feature", "pair_id"]) == []
    assert future_label_like_columns(["h900_origin_reach_75"]) == ["h900_origin_reach_75"]

    # Synthetic deterministic matching with destruction included.
    rows = []
    for i in range(30):
        rows.append({
            "candidate_id": f"H{i:03d}", "candidate_date": "2026-06-01", "direction": 1,
            "__oriented__x": 2.0 + i / 1000, "__control_momentum": i / 100,
            "__control_aggression": i / 90, "__control_destruction": i / 80,
        })
        rows.append({
            "candidate_id": f"L{i:03d}", "candidate_date": "2026-06-01", "direction": 1,
            "__oriented__x": -2.0 - i / 1000, "__control_momentum": i / 100 + 0.002,
            "__control_aggression": i / 90 + 0.002, "__control_destruction": i / 80 + 0.002,
        })
    d = pd.DataFrame(rows)
    d = add_z_controls(d, PRIMARY_CONTROL_COLUMNS)
    high = d[d["candidate_id"].str.startswith("H")]
    low = d[d["candidate_id"].str.startswith("L")]
    p = greedy_match(high, low, feature="x", control_tier=PRIMARY_CONTROL_TIER, controls=PRIMARY_CONTROL_COLUMNS, population="ALL", observation_horizon_sec=3)
    assert len(p) == 30
    b = pair_balance(p, PRIMARY_CONTROL_COLUMNS)
    assert safe_float(b["max_abs_smd"]) < 0.2

    # Many-to-one outcome join across repeated pair sets is legitimate.
    repeated = pd.concat([p.iloc[:3], p.iloc[:3]], ignore_index=True)
    repeated["observation_horizon_sec"] = 3
    out_ids = sorted(set(repeated["high_candidate_id"]) | set(repeated["low_candidate_id"]))
    outcomes = pd.DataFrame({
        "candidate_id": out_ids,
        "observation_horizon_sec": 3,
        "decision_price_valid": True,
        "h300_fully_observed": True, "h600_fully_observed": True, "h900_fully_observed": True, "h1800_fully_observed": True,
        "h900_origin_reach_50": False,
        "h900_origin_reach_75": [i % 2 == 0 for i in range(len(out_ids))],
        "h900_origin_reach_100": False,
    })
    joined = pair_outcome_frame(repeated, outcomes)
    assert len(joined) == len(repeated)
    print("SELF-TEST PASS: QC label detector, destruction-matched matcher, balance, and repeated many-to-one outcome join validated.")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.5 prospective 25-bp opposing refill-fate attribution")
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--m4-root", type=Path, default=DEFAULT_M4_ROOT, help="Optional 7M.4 root used only for validated exact-static causal cache")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    require_stack()
    built_at = utc_now_iso()

    print("Phase 7M.5 — Prospective 25-bp opposing refill-fate attribution", flush=True)
    print("NO raw replay | NO candidate regeneration | NO ML | NO composite score", flush=True)
    print(f"Primary control tier   : {PRIMARY_CONTROL_TIER}", flush=True)
    print(f"Destruction control    : {DESTRUCTION_ANCHOR}", flush=True)
    print("Primary outcome        : origin reaches 75bp within 15m", flush=True)

    print("[1/9] Loading and validating frozen Phase 7M.3 causal substrate...", flush=True)
    m3_meta, candidates, mechanics, outcome_path = load_frozen_inputs(args.m3_root)
    candidate_ids = set(candidates["candidate_id"].astype(str))
    candidate_hash = sha256_ids(candidate_ids)
    print(f"      candidates={len(candidate_ids):,}; mechanics={len(mechanics):,}; id_hash={candidate_hash[:16]}...", flush=True)

    print("[2/9] Loading exact initial-book state (validated 7M.4 cache if available)...", flush=True)
    static_book, static_summary = load_or_build_static_book(args.m3_root, args.m4_root, candidate_ids)
    print(
        f"      source={static_summary.get('source')} coverage={100*safe_float(static_summary.get('exact_static_coverage')):.3f}%",
        flush=True,
    )

    print("[3/9] Building FUTURE-CLEAN causal analysis inputs...", flush=True)
    causal = build_causal_inputs(candidates, mechanics, static_book)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_parquet(causal, args.output_root / "analysis_inputs.parquet")
    atomic_parquet(static_book, args.output_root / "exact_initial_book_state.parquet")

    print("[4/9] OUTCOME-BLIND refill-fate physical audit...", flush=True)
    audit, destruction_terciles = refill_fate_physical_audit(causal)
    atomic_csv(audit, args.report_root / "refill_fate_physical_audit.csv")
    atomic_csv(destruction_terciles, args.report_root / "refill_fate_by_destruction_tercile.csv")
    for _, r in audit.iterrows():
        print(
            f"      {r['feature'][:48]:48s} observed={100*safe_float(r['observed_fraction']):5.1f}% "
            f"rho(destruction)={safe_float(r['destruction_to_oriented_spearman']):+.3f}",
            flush=True,
        )

    print("[5/9] PRE-OUTCOME destruction-matched feasibility and balance...", flush=True)
    pairs, gate = build_pair_sets(causal)
    # Freeze pair manifests BEFORE any outcome file is read.
    atomic_parquet(pairs, args.output_root / "matched_pairs.parquet")
    atomic_csv(gate, args.report_root / "precompute_feasibility_gate.csv")
    atomic_csv(gate, args.report_root / "matching_balance.csv")
    pg = gate[
        (pd.to_numeric(gate["observation_horizon_sec"], errors="coerce") == PRIMARY_OBSERVATION_HORIZON_SEC)
        & (gate["population"] == "ALL")
    ]
    for _, r in pg.iterrows():
        print(
            f"      {r['control_tier']:19s} {r['feature'][:42]:42s} pairs={int(r['pairs_after_repair']):4d} "
            f"D/V={int(r['discovery_pairs'])}/{int(r['validation_pairs'])} max|SMD|={safe_float(r['post_max_abs_smd']):.3f} "
            f"FEASIBLE={bool(r['feasible_before_label_load'])}", flush=True,
        )

    print("[6/9] Causal manifests frozen. Loading future outcomes NOW...", flush=True)
    outcomes = load_outcomes_after_freeze(outcome_path)
    joined = pair_outcome_frame(pairs, outcomes)
    print(f"      pair rows={len(pairs):,}; outcome-joined rows={len(joined):,}", flush=True)

    print("[7/9] Estimating preregistered primary and robustness contrasts...", flush=True)
    primary, split, static = analyze_primary(joined, gate, int(args.bootstrap_reps))
    obs_robust = analyze_observation_horizons(joined, gate)
    outcome_robust = analyze_outcome_horizons(joined, gate)
    direction = analyze_directions(joined)
    reaccel = analyze_reaccel(joined, gate, int(args.bootstrap_reps))
    day_sign = day_sign_consistency(joined)

    atomic_csv(primary, args.report_root / "primary_refill_fate_contrasts.csv")
    atomic_csv(split, args.report_root / "discovery_validation.csv")
    atomic_csv(static, args.report_root / "static_state_robustness.csv")
    atomic_csv(obs_robust, args.report_root / "observation_horizon_robustness.csv")
    atomic_csv(outcome_robust, args.report_root / "outcome_horizon_robustness.csv")
    atomic_csv(direction, args.report_root / "direction_robustness.csv")
    atomic_csv(day_sign, args.report_root / "feature_day_sign_consistency.csv")
    atomic_csv(reaccel, args.report_root / "reacceleration_diagnostic.csv")

    print("[8/9] Checking invariants and causal/future separation...", flush=True)
    invariants = build_invariants(
        causal=causal, pairs=pairs, gate=gate, outcomes=outcomes,
        static_book=static_book, m3_meta=m3_meta, audit=audit,
    )
    atomic_csv(invariants, args.report_root / "analysis_invariants.csv")
    failed = invariants[invariants["status"] == "FAIL"]
    for _, r in invariants.iterrows():
        print(f"      {r['status']:4s} {r['invariant']}: observed={r['observed']} expected={r['expected']}", flush=True)

    print("[9/9] Writing metadata and mechanical findings...", flush=True)
    status = "PASS" if failed.empty else "FAIL"
    p75 = primary[primary["target_bps"] == PRIMARY_TARGET_BPS] if not primary.empty else primary
    supported_primary = int((p75["interpretation"] == "SUPPORTED").sum()) if not p75.empty else 0
    s75 = static[static["target_bps"] == PRIMARY_TARGET_BPS] if not static.empty else static
    supported_static = int((s75["interpretation"] == "SUPPORTED").sum()) if not s75.empty else 0

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "built_at_utc": built_at,
        "status": status,
        "scientific_state": "PROSPECTIVE_25BP_OPPOSING_REFILL_FATE_ATTRIBUTION_COMPLETE_NOT_ALPHA",
        "candidate_count": EXPECTED_CANDIDATES,
        "candidate_id_set_sha256": candidate_hash,
        "frozen_phase7m3_builder": FROZEN_PHASE7M3_BUILDER,
        "frozen_phase7m3_schema": FROZEN_PHASE7M3_SCHEMA,
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "destruction_anchor": DESTRUCTION_ANCHOR,
        "primary_control_tier": PRIMARY_CONTROL_TIER,
        "strict_control_tier": STRICT_CONTROL_TIER,
        "raw_depth_replay": False,
        "raw_trade_replay": False,
        "candidate_regeneration": False,
        "ml": False,
        "composite_score": False,
        "future_labels_loaded_only_after_causal_pair_manifest_written": True,
        "refill_fate_features_tested": [x[0] for x in REFILL_FATE_FEATURES],
        "supported_primary_p75_features": supported_primary,
        "supported_strict_static_p75_features": supported_static,
        "failed_invariants": int(len(failed)),
    }
    config = {
        **summary,
        "m3_root": str(args.m3_root),
        "m4_root": str(args.m4_root),
        "output_root": str(args.output_root),
        "report_root": str(args.report_root),
        "feature_contract": [
            {"feature": f, "orientation_sign": sign, "family": family, "oriented_meaning": meaning}
            for f, sign, family, meaning in REFILL_FATE_FEATURES
        ],
        "matching": {
            "exact_blocks": ["candidate_date", "direction"],
            "primary_controls": list(PRIMARY_CONTROL_COLUMNS),
            "strict_controls": list(STRICT_CONTROL_COLUMNS),
            "z_caliper": MATCH_Z_CALIPER,
            "euclidean_caliper": MATCH_EUCLIDEAN_CALIPER,
            "replacement": False,
            "balance_repair": "OUTCOME_BLIND_DETERMINISTIC_PAIR_TRIM",
            "balance_smd_gate": BALANCE_SMD_GATE,
        },
        "min_full_pairs": MIN_FULL_PAIRS,
        "min_split_pairs": MIN_SPLIT_PAIRS,
        "min_reaccel_pairs": MIN_REACCEL_PAIRS,
        "discovery_end_date": DISCOVERY_END_DATE,
        "validation_start_date": VALIDATION_START_DATE,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "exact_static_state": static_summary,
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_json(args.output_root / "metadata" / "run_config.json", config)

    findings = make_findings(
        audit=audit, gate=gate, primary=primary, split=split, static=static,
        obs_robust=obs_robust, outcome_robust=outcome_robust,
        reaccel=reaccel, invariants=invariants,
    )
    atomic_text(args.report_root / "findings.md", findings)

    print("", flush=True)
    print(f"Phase 7M.5 status: {status}", flush=True)
    print(f"Primary P75 supported features: {supported_primary}/{len(REFILL_FATE_FEATURES)}", flush=True)
    print(f"Strict-static P75 supported features: {supported_static}/{len(REFILL_FATE_FEATURES)}", flush=True)
    print(f"Reports: {args.report_root}", flush=True)
    print(f"Data:    {args.output_root}", flush=True)
    if not failed.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
