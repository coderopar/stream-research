#!/usr/bin/env python3
"""
Phase 7M.4 — Prospective 25-bp exact mechanism attribution
==========================================================

PLAIN-ENGLISH QUESTION
----------------------
Phase 7M.2 established that, at a genuinely prospective 25-bp candidate,
unusually strong opposing-side withdrawal during the next three seconds is
associated with a higher chance of eventually reaching 75 bps.

Phase 7M.3 then replayed the exact authoritative book/trade stream for the full
prospective 25-bp population, including attempts that later fail before 35 bps.

Phase 7M.4 asks:

    WHAT exact queue-level behavior is the cheap 25-bp withdrawal signal
    actually capturing?

and:

    Does that exact behavior itself distinguish future 75-bp continuation
    after controlling for immediate momentum and executed aggression?

This is a CHEAP ANALYSIS-ONLY phase.

NO raw depth replay.
NO raw trade replay.
NO candidate regeneration.
NO ML.
NO composite book-failure score.
NO profitability claim.

SCIENTIFIC SEQUENCE
-------------------
The phase deliberately separates causal construction from future outcomes.

STEP A — OUTCOME-BLIND MECHANISM BRIDGE
    Join the Phase-7M.2 cheap 25-bp opposing-withdrawal proxy to the frozen
    Phase-7M.3 exact 3-second mechanics.

    Ask whether high cheap withdrawal corresponds physically to:
      * greater exact opposing initial-queue loss;
      * lower retained initial opposing liquidity;
      * greater durable unrecovered opposing loss;
      * weaker durable same-price refill;
      * stronger support-vs-opposing resilience.

    NO continuation labels are loaded for this step.

STEP B — OUTCOME-BLIND MATCHING / FEASIBILITY
    For each PREDECLARED exact mechanism, form high-failure versus low-failure
    terciles and match candidates one-to-one without replacement.

    Exact matching blocks:
      candidate_date
      direction

    BASE continuous controls:
      3-second directional price momentum
      log1p(3-second directional aggressive quote)

    STRICT_STATIC continuous controls:
      BASE controls
      log1p(exact initial opposing frozen 5-bp quote)
      log1p(exact initial support frozen 5-bp quote)

    The exact initial book controls are reconstructed only from Phase-7M.3's
    frozen level_audit parquet checkpoints:
        initial quote = price * initial_qty
    summed separately for SUPPORT and OPPOSING levels.

    No Phase-6B static-depth substitute is silently used.

    Deterministic matching:
      * 1:1
      * no replacement
      * exact date + direction
      * per-control |z difference| <= 0.75
      * Euclidean z distance <= 1.00

    Outcome-blind balance repair:
      * every control |SMD| <= 0.20

PRE-COMPUTE FEASIBILITY GATES
-----------------------------
FULL population / mechanism / primary 3s comparison:
    >= 250 balanced pairs

Discovery and validation:
    >= 100 pairs each

Causal re-acceleration diagnostic:
    >= 75 balanced pairs

If a test fails the pre-outcome gate:
    report NOT TESTED
    do not loosen the matcher after seeing outcomes.

STEP C — ONLY AFTER PAIRS ARE FROZEN, LOAD OUTCOMES
Primary:
    origin reaches 75 bps within 15 minutes of the +3s decision timestamp

Secondary:
    origin reaches 50 bps within 15 minutes
    origin reaches 100 bps within 15 minutes
    origin reaches 35 bps within 15 minutes IF that frozen outcome field exists

Robustness:
    75 bps within 5 / 10 / 30 minutes
    same exact mechanism at 1 / 5 / 10-second observation horizons

Chronological split:
    DISCOVERY  candidate_date <= 2026-06-17
    VALIDATION candidate_date >= 2026-06-18

Uncertainty:
    candidate-day cluster bootstrap over matched-pair outcome differences.

PREDECLARED EXACT FEATURES
--------------------------
Higher ORIENTED value always means the hypothesized continuation-favoring
liquidity-failure/support state.

CORE OPPOSING FAILURE:
    + opposing__durable_unrecovered_loss_share_initial
    + opposing__peak_initial_queue_loss_share
    - opposing__retained_initial_quote_ratio
    + support_minus_opposing_retained_ratio

REFILL / DURABILITY CONTEXT:
    - opposing__same_price_refill_to_depletion_ratio
    - opposing__durable_same_price_refill_fraction

SUPPORT CONTEXT:
    + support__retained_initial_quote_ratio
    + support__same_price_refill_to_depletion_ratio

Important dependence:
    opposing retained initial quote and durable unrecovered opposing loss are
    mechanically close descriptions of the same queue-survival process.
    They are NOT counted as independent biological/market mechanisms merely
    because both are statistically supported.

PRIMARY OBSERVATION HORIZON
---------------------------
3 seconds, frozen in advance.

1 / 5 / 10 seconds are robustness only.
No horizon is selected based on outcomes.

PRIMARY OUTCOME
---------------
75 bps within 15 minutes from the exact +3-second decision timestamp.

DISCOVERY / VALIDATION
----------------------
Same split used in earlier threshold work:

    DISCOVERY  <= 2026-06-17
    VALIDATION >= 2026-06-18

A feature is not called chronologically replicated merely because the pooled
confidence interval is positive.  Discovery and validation effect directions
must both be positive.

INTERPRETATION
--------------
If an exact dynamic mechanism remains positive after STRICT_STATIC matching,
that is evidence that the important information is not simply:

    "the book started thin"

but instead something closer to:

    "the opposing book dynamically fails to retain/restabilize liquidity
     under pressure."

If the BASE effect disappears after exact static-state control, interpretation
must be reduced toward starting-depth state.

If the cheap withdrawal proxy is predictive but does not bridge to the
predeclared exact mechanisms, STOP and reconsider proxy semantics rather than
inventing a new exact feature after seeing outcomes.

INPUTS
------
Frozen Phase 7M.3:
data/derived/perps/research/prospective_25bp_exact_participant_interaction_v1/
    candidate_inputs.parquet
    candidate_mechanics.parquet
    candidate_decision_outcomes.parquet
    level_audit/candidate_date=*/part.parquet
    metadata/latest_summary.json

Frozen Phase 7M.2:
data/derived/perps/research/prospective_early_candidate_audit_v1/
    early_candidate_inputs.parquet
    metadata/latest_summary.json

OUTPUTS
-------
data/derived/perps/research/prospective_25bp_exact_mechanism_analysis_v1/
    analysis_inputs.parquet
    matched_pairs.parquet
    exact_initial_book_state.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_25bp_exact_mechanism_analysis/
    mechanism_bridge.csv
    mechanism_bridge_by_withdrawal_tercile.csv
    precompute_feasibility_gate.csv
    matching_balance.csv
    primary_exact_mechanism_contrasts.csv
    discovery_validation.csv
    static_state_robustness.csv
    observation_horizon_robustness.csv
    outcome_horizon_robustness.csv
    direction_robustness.csv
    reacceleration_diagnostic.csv
    feature_day_sign_consistency.csv
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


BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = 1
SCHEMA_NAME = "prospective_25bp_exact_mechanism_analysis_v1"

FROZEN_PHASE7M3_BUILDER = "1.0.0"
FROZEN_PHASE7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"
FROZEN_PHASE7M2_BUILDER = "1.0.0"
FROZEN_PHASE7M2_SCHEMA = "prospective_early_candidate_audit_v1"

FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"
EXPECTED_CANDIDATES = 7_067
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_OBSERVATION_HORIZON_SEC = 3
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
PRIMARY_OUTCOME_HORIZON_SEC = 900
PRIMARY_TARGET_BPS = 75

DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

CHEAP_WITHDRAWAL_FEATURE = "opposing__peak_1s_withdrawal_frac_3s"

# Higher oriented value = hypothesized continuation-favoring exact state.
BOOK_FEATURES: tuple[tuple[str, int, str], ...] = (
    (
        "opposing__durable_unrecovered_loss_share_initial",
        +1,
        "CORE_OPPOSING_FAILURE",
    ),
    (
        "opposing__peak_initial_queue_loss_share",
        +1,
        "CORE_OPPOSING_FAILURE",
    ),
    (
        "opposing__retained_initial_quote_ratio",
        -1,
        "CORE_OPPOSING_FAILURE",
    ),
    (
        "support_minus_opposing_retained_ratio",
        +1,
        "CORE_OPPOSING_FAILURE",
    ),
    (
        "opposing__same_price_refill_to_depletion_ratio",
        -1,
        "REFILL_DURABILITY_CONTEXT",
    ),
    (
        "opposing__durable_same_price_refill_fraction",
        -1,
        "REFILL_DURABILITY_CONTEXT",
    ),
    (
        "support__retained_initial_quote_ratio",
        +1,
        "SUPPORT_CONTEXT",
    ),
    (
        "support__same_price_refill_to_depletion_ratio",
        +1,
        "SUPPORT_CONTEXT",
    ),
)
CORE_BOOK_FEATURES = tuple(
    x[0] for x in BOOK_FEATURES if x[2] == "CORE_OPPOSING_FAILURE"
)

PRIMARY_MOMENTUM_CONTROL = "directional_mid_move_bps"
PRIMARY_AGGRESSION_CONTROL = "directional_aggressive_quote"

BASE_CONTROL_COLUMNS = (
    "__control_momentum",
    "__control_aggression",
)
STRICT_CONTROL_COLUMNS = (
    "__control_momentum",
    "__control_aggression",
    "__control_initial_opposing",
    "__control_initial_support",
)

BOOK_BINS = 3
WITHDRAWAL_BINS = 3
MATCH_Z_CALIPER = 0.75
MATCH_EUCLIDEAN_CALIPER = 1.00
BALANCE_SMD_GATE = 0.20

MIN_FULL_PAIRS = 250
MIN_SPLIT_PAIRS = 100
MIN_REACCEL_PAIRS = 75
MIN_DIRECTION_PAIRS = 100

BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260831
MIN_BOOTSTRAP_VALID_REPS = 250

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = (
    SCRIPT_PATH.parents[3]
    if len(SCRIPT_PATH.parents) >= 4
    else Path.cwd()
)

DEFAULT_M3_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "perps"
    / "research"
    / FROZEN_PHASE7M3_SCHEMA
)
DEFAULT_M2_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "perps"
    / "research"
    / FROZEN_PHASE7M2_SCHEMA
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "perps"
    / "research"
    / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT
    / "reports"
    / "perps_prospective_25bp_exact_mechanism_analysis"
)


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
        raise SystemExit(
            "Phase 7M.4 requires pandas, numpy and pyarrow in the project .venv."
        )


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
    return hashlib.sha256(
        "\n".join(sorted(str(x) for x in ids)).encode("utf-8")
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m4-",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m4-",
        suffix=".csv",
        dir=str(path.parent),
    )
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
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m4-",
        suffix=".parquet",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            tmp,
            compression="zstd",
        )
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_metadata(
    root: Path,
    expected_builder: str,
    expected_schema: str,
    label: str,
) -> dict[str, Any]:
    p = root / "metadata" / "latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing {label} metadata: {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if str(meta.get("builder_version")) != str(expected_builder):
        raise RuntimeError(
            f"{label} builder mismatch: expected {expected_builder}, "
            f"found {meta.get('builder_version')!r}"
        )
    if str(meta.get("schema_name")) != str(expected_schema):
        raise RuntimeError(
            f"{label} schema mismatch: expected {expected_schema}, "
            f"found {meta.get('schema_name')!r}"
        )
    if str(meta.get("status")) != "PASS":
        raise RuntimeError(
            f"{label} must be PASS; found status={meta.get('status')!r}"
        )
    return meta


def percentile_bin(s: pd.Series, bins: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if not mask.any():
        return out
    ranks = x.loc[mask].rank(method="first", pct=True)
    out.loc[mask] = (
        np.ceil(ranks * bins)
        .clip(1, bins)
        .astype(int)
        .astype("Int64")
    )
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
    xr = x.loc[mask].rank(method="average")
    yr = y.loc[mask].rank(method="average")
    return safe_float(xr.corr(yr))


def robust_scale_diff(a: pd.Series, b: pd.Series) -> float:
    """Median difference divided by pooled IQR-like robust scale."""
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if len(x) < 3 or len(y) < 3:
        return math.nan
    scale_x = (x.quantile(0.75) - x.quantile(0.25)) / 1.349
    scale_y = (y.quantile(0.75) - y.quantile(0.25)) / 1.349
    den = math.sqrt(0.5 * (scale_x * scale_x + scale_y * scale_y))
    if not math.isfinite(den) or den <= 1e-12:
        return math.nan
    return safe_float((x.median() - y.median()) / den)


def load_frozen_inputs(m3_root: Path, m2_root: Path):
    require_stack()

    m3_meta = load_metadata(
        m3_root,
        FROZEN_PHASE7M3_BUILDER,
        FROZEN_PHASE7M3_SCHEMA,
        "Phase 7M.3",
    )
    m2_meta = load_metadata(
        m2_root,
        FROZEN_PHASE7M2_BUILDER,
        FROZEN_PHASE7M2_SCHEMA,
        "Phase 7M.2",
    )

    if str(m3_meta.get("run_scope")) != "FULL":
        raise RuntimeError(
            f"Phase 7M.3 must be FULL; found {m3_meta.get('run_scope')!r}"
        )
    if int(m3_meta.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} frozen Phase7M.3 candidates; "
            f"metadata reports {m3_meta.get('candidate_count')!r}"
        )
    if safe_float(m3_meta.get("candidate_threshold_bps")) != 25.0:
        raise RuntimeError(
            "Phase 7M.3 candidate threshold must remain frozen at 25 bps."
        )
    if not bool(m3_meta.get("prospective_population_includes_failed_before35")):
        raise RuntimeError(
            "Phase 7M.3 metadata does not confirm fail-before-35 inclusion."
        )

    mech_path = m3_root / "candidate_mechanics.parquet"
    input_path = m3_root / "candidate_inputs.parquet"
    outcome_path = m3_root / "candidate_decision_outcomes.parquet"
    cheap_path = m2_root / "early_candidate_inputs.parquet"

    for p in (mech_path, input_path, outcome_path, cheap_path):
        if not p.exists():
            raise RuntimeError(f"Missing frozen input: {p}")

    mechanics = pq.read_table(mech_path).to_pandas()
    candidate_inputs = pq.read_table(input_path).to_pandas()
    cheap_inputs = pq.read_table(cheap_path).to_pandas()

    required_mech = [
        "candidate_id",
        "candidate_date",
        "direction",
        "direction_label",
        "observation_horizon_sec",
        "window_valid",
        "decision_price_valid",
        PRIMARY_MOMENTUM_CONTROL,
        PRIMARY_AGGRESSION_CONTROL,
        "extra_displacement_since_candidate_bps",
    ] + [x[0] for x in BOOK_FEATURES]
    require_columns(mechanics, required_mech, "Phase7M.3 candidate_mechanics")

    required_candidates = [
        "candidate_id",
        "candidate_date",
        "direction",
        "policy_id",
        "candidate_threshold_bps",
    ]
    require_columns(candidate_inputs, required_candidates, "Phase7M.3 candidate_inputs")

    require_columns(
        cheap_inputs,
        [
            "candidate_id",
            "early_threshold_bps",
            "proxy_valid",
            CHEAP_WITHDRAWAL_FEATURE,
        ],
        "Phase7M.2 early_candidate_inputs",
    )

    candidate_inputs = candidate_inputs[
        candidate_inputs["policy_id"].astype(str) == FROZEN_POLICY
    ].copy()
    cheap_inputs = cheap_inputs[
        pd.to_numeric(cheap_inputs["early_threshold_bps"], errors="coerce") == 25
    ].copy()

    if candidate_inputs["candidate_id"].duplicated().any():
        raise RuntimeError("Phase7M.3 candidate_inputs duplicate candidate IDs.")
    if len(candidate_inputs) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} candidate inputs; "
            f"found {len(candidate_inputs):,}"
        )

    horizons = tuple(
        sorted(
            pd.to_numeric(
                mechanics["observation_horizon_sec"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )
    )
    if horizons != EXPECTED_HORIZONS:
        raise RuntimeError(
            f"Phase7M.3 mechanics horizons mismatch: {horizons}"
        )
    expected_rows = EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS)
    if len(mechanics) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} mechanics rows; found {len(mechanics):,}"
        )
    if mechanics.duplicated(
        ["candidate_id", "observation_horizon_sec"]
    ).any():
        raise RuntimeError(
            "Phase7M.3 mechanics duplicate candidate/horizon keys."
        )

    return (
        m3_meta,
        m2_meta,
        candidate_inputs,
        mechanics,
        cheap_inputs,
        outcome_path,
    )


def load_exact_initial_book_state(
    m3_root: Path,
    candidate_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Derive exact starting support/opposing frozen 5-bp quote from level audit.

    No raw replay:
      initial_quote = price * initial_qty
    summed per candidate and side_role.
    """
    require_stack()

    paths = sorted(
        (m3_root / "level_audit").glob("candidate_date=*/part.parquet")
    )
    if not paths:
        raise RuntimeError(
            "No frozen Phase7M.3 level_audit date partitions found."
        )

    parts = []
    key_name_seen: set[str] = set()
    rows_read = 0

    for i, path in enumerate(paths, 1):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        key = "candidate_id" if "candidate_id" in names else (
            "event_id" if "event_id" in names else None
        )
        if key is None:
            raise RuntimeError(
                f"{path} lacks candidate_id/event_id in level audit schema."
            )
        required = {key, "side_role", "price", "initial_qty"}
        missing = required - names
        if missing:
            raise RuntimeError(
                f"{path} missing exact starting-book fields: {sorted(missing)}"
            )

        d = pq.read_table(
            path,
            columns=[key, "side_role", "price", "initial_qty"],
        ).to_pandas()
        rows_read += len(d)
        key_name_seen.add(key)

        d["candidate_id"] = d[key].astype(str)
        d = d[d["candidate_id"].isin(candidate_ids)].copy()
        if d.empty:
            continue

        role = d["side_role"].astype(str).str.upper()
        if not set(role.dropna().unique()).issubset({"SUPPORT", "OPPOSING"}):
            raise RuntimeError(
                f"Unexpected exact level roles in {path}: "
                f"{sorted(set(role.dropna().unique()))}"
            )
        d["side_role"] = role
        price = pd.to_numeric(d["price"], errors="coerce")
        qty = pd.to_numeric(d["initial_qty"], errors="coerce")
        if ((qty <= 0) & qty.notna()).any():
            raise RuntimeError(
                f"Nonpositive exact initial_qty found in {path}."
            )
        d["__initial_quote"] = price * qty
        d = d[
            d["__initial_quote"].notna()
            & np.isfinite(d["__initial_quote"])
            & (d["__initial_quote"] > 0)
        ].copy()

        g = (
            d.groupby(["candidate_id", "side_role"])["__initial_quote"]
            .agg(["sum", "size"])
            .reset_index()
        )
        parts.append(g)

        if i == 1 or i % 10 == 0 or i == len(paths):
            print(
                f"      exact static book [{i:02d}/{len(paths):02d}] "
                f"{path.parent.name}",
                flush=True,
            )

    if not parts:
        raise RuntimeError("Exact level audit produced no candidate book state.")

    agg = pd.concat(parts, ignore_index=True)
    agg = (
        agg.groupby(["candidate_id", "side_role"], as_index=False)
        .agg(initial_quote=("sum", "sum"), initial_levels=("size", "sum"))
    )

    quote = agg.pivot(
        index="candidate_id",
        columns="side_role",
        values="initial_quote",
    )
    levels = agg.pivot(
        index="candidate_id",
        columns="side_role",
        values="initial_levels",
    )

    out = pd.DataFrame(index=sorted(candidate_ids))
    out.index.name = "candidate_id"
    out["initial_support_quote_5bps_exact"] = quote.get("SUPPORT")
    out["initial_opposing_quote_5bps_exact"] = quote.get("OPPOSING")
    out["initial_support_levels_5bps_exact"] = levels.get("SUPPORT")
    out["initial_opposing_levels_5bps_exact"] = levels.get("OPPOSING")
    out = out.reset_index()

    out["initial_support_minus_opposing_log_quote"] = (
        np.log1p(
            pd.to_numeric(
                out["initial_support_quote_5bps_exact"], errors="coerce"
            )
        )
        - np.log1p(
            pd.to_numeric(
                out["initial_opposing_quote_5bps_exact"], errors="coerce"
            )
        )
    )

    both = (
        pd.to_numeric(
            out["initial_support_quote_5bps_exact"], errors="coerce"
        ).notna()
        & pd.to_numeric(
            out["initial_opposing_quote_5bps_exact"], errors="coerce"
        ).notna()
    )
    summary = {
        "level_partitions": len(paths),
        "level_rows_read": rows_read,
        "level_id_columns_seen": sorted(key_name_seen),
        "candidate_rows": len(out),
        "candidates_with_both_exact_sides": int(both.sum()),
        "exact_static_coverage": float(both.mean()) if len(out) else math.nan,
    }
    return out, summary


def build_causal_analysis_inputs(
    candidate_inputs: pd.DataFrame,
    mechanics: pd.DataFrame,
    cheap_inputs: pd.DataFrame,
    static_book: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outcome-free analysis table.

    The result can be written to disk before candidate_decision_outcomes is read.
    """
    candidate_cols = [
        "candidate_id",
        "candidate_date",
        "direction",
        "direction_label",
        "policy_id",
        "candidate_threshold_bps",
    ]
    for optional in ("origin_kind", "is_reaccel_context"):
        if optional in candidate_inputs.columns:
            candidate_cols.append(optional)

    candidates = candidate_inputs[candidate_cols].copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    cheap = cheap_inputs[
        [
            "candidate_id",
            "proxy_valid",
            CHEAP_WITHDRAWAL_FEATURE,
        ]
    ].copy()
    cheap["candidate_id"] = cheap["candidate_id"].astype(str)

    m = mechanics.copy()
    m["candidate_id"] = m["candidate_id"].astype(str)

    causal = (
        m.merge(
            candidates,
            on=[
                "candidate_id",
                "candidate_date",
                "direction",
                "direction_label",
            ],
            how="left",
            validate="many_to_one",
            suffixes=("", "__candidate"),
        )
        .merge(
            cheap,
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            static_book,
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
    )

    # Explicit causal controls.
    causal["__control_momentum"] = pd.to_numeric(
        causal[PRIMARY_MOMENTUM_CONTROL], errors="coerce"
    )
    causal["__control_aggression"] = np.log1p(
        pd.to_numeric(
            causal[PRIMARY_AGGRESSION_CONTROL], errors="coerce"
        ).clip(lower=0.0)
    )
    causal["__control_initial_opposing"] = np.log1p(
        pd.to_numeric(
            causal["initial_opposing_quote_5bps_exact"], errors="coerce"
        )
    )
    causal["__control_initial_support"] = np.log1p(
        pd.to_numeric(
            causal["initial_support_quote_5bps_exact"], errors="coerce"
        )
    )

    # Outcome-blind orientation/bins separately per observation horizon.
    for h in EXPECTED_HORIZONS:
        hm = (
            pd.to_numeric(
                causal["observation_horizon_sec"], errors="coerce"
            ) == h
        )
        for feature, sign, _family in BOOK_FEATURES:
            oriented = (
                pd.to_numeric(causal.loc[hm, feature], errors="coerce")
                * float(sign)
            )
            causal.loc[hm, f"__oriented__{feature}"] = oriented
            causal.loc[hm, f"__book_tercile__{feature}"] = (
                percentile_bin(oriented, BOOK_BINS).astype(float)
            )

        if h == PRIMARY_OBSERVATION_HORIZON_SEC:
            withdrawal = pd.to_numeric(
                causal.loc[hm, CHEAP_WITHDRAWAL_FEATURE], errors="coerce"
            )
            causal.loc[hm, "__cheap_withdrawal_tercile"] = (
                percentile_bin(withdrawal, WITHDRAWAL_BINS).astype(float)
            )

    # This table MUST remain future-clean.
    forbidden = [
        c
        for c in causal.columns
        if (
            c.startswith("h300_")
            or c.startswith("h600_")
            or c.startswith("h900_")
            or c.startswith("h1800_")
            or "origin_reach_" in c.lower()
            or "future_outcome" in c.lower()
        )
    ]
    if forbidden:
        raise RuntimeError(
            f"Future labels leaked into causal analysis inputs: {forbidden}"
        )
    return causal


def mechanism_bridge(causal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outcome-blind cheap-withdrawal -> exact-mechanics bridge at 3 seconds.
    """
    d = causal[
        (
            pd.to_numeric(
                causal["observation_horizon_sec"], errors="coerce"
            )
            == PRIMARY_OBSERVATION_HORIZON_SEC
        )
        & bool_series(causal["window_valid"])
        & bool_series(causal["proxy_valid"])
    ].copy()

    bridge_rows = []
    tercile_rows = []

    for feature, sign, family in BOOK_FEATURES:
        oriented_col = f"__oriented__{feature}"
        x = d[
            pd.to_numeric(
                d[CHEAP_WITHDRAWAL_FEATURE], errors="coerce"
            ).notna()
            & pd.to_numeric(d[oriented_col], errors="coerce").notna()
        ].copy()

        corr = spearman_rank_corr(
            x[CHEAP_WITHDRAWAL_FEATURE],
            x[oriented_col],
        )

        wt = percentile_bin(
            x[CHEAP_WITHDRAWAL_FEATURE],
            WITHDRAWAL_BINS,
        )
        hi = x.loc[pd.to_numeric(wt, errors="coerce") == WITHDRAWAL_BINS]
        lo = x.loc[pd.to_numeric(wt, errors="coerce") == 1]

        bridge_rows.append(
            {
                "feature": feature,
                "feature_family": family,
                "feature_orientation_sign": sign,
                "paired_rows": int(len(x)),
                "cheap_withdrawal_to_oriented_exact_spearman": corr,
                "high_withdrawal_n": int(len(hi)),
                "low_withdrawal_n": int(len(lo)),
                "high_withdrawal_exact_oriented_mean": safe_float(
                    hi[oriented_col].mean()
                ),
                "low_withdrawal_exact_oriented_mean": safe_float(
                    lo[oriented_col].mean()
                ),
                "high_minus_low_exact_oriented_mean": safe_float(
                    hi[oriented_col].mean() - lo[oriented_col].mean()
                ),
                "high_withdrawal_exact_oriented_median": safe_float(
                    hi[oriented_col].median()
                ),
                "low_withdrawal_exact_oriented_median": safe_float(
                    lo[oriented_col].median()
                ),
                "robust_standardized_high_minus_low": robust_scale_diff(
                    hi[oriented_col], lo[oriented_col]
                ),
            }
        )

        for tercile in (1, 2, 3):
            t = x.loc[pd.to_numeric(wt, errors="coerce") == tercile]
            tercile_rows.append(
                {
                    "feature": feature,
                    "feature_family": family,
                    "cheap_withdrawal_tercile": tercile,
                    "rows": int(len(t)),
                    "cheap_withdrawal_median": safe_float(
                        t[CHEAP_WITHDRAWAL_FEATURE].median()
                    ),
                    "exact_oriented_mean": safe_float(
                        t[oriented_col].mean()
                    ),
                    "exact_oriented_median": safe_float(
                        t[oriented_col].median()
                    ),
                }
            )

    return pd.DataFrame(bridge_rows), pd.DataFrame(tercile_rows)


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
    """
    Deterministic exact-date/direction 1:1 greedy matching without replacement.
    """
    rows = []

    for (candidate_date, direction), hb in high.groupby(
        ["candidate_date", "direction"],
        sort=True,
    ):
        lb = low[
            (low["candidate_date"].astype(str) == str(candidate_date))
            & (
                pd.to_numeric(low["direction"], errors="coerce")
                == int(direction)
            )
        ].copy()

        if hb.empty or lb.empty:
            continue

        # Most failure-oriented high rows first; stable candidate-id tie break.
        hb = hb.sort_values(
            [f"__oriented__{feature}", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        lb = lb.sort_values("candidate_id", kind="mergesort")

        available = set(lb.index.tolist())

        for hi_idx, hr in hb.iterrows():
            best_idx = None
            best_key = None
            best_diffs = None

            for lo_idx in sorted(available):
                lr = lb.loc[lo_idx]
                diffs = []
                valid = True
                for c in controls:
                    zc = f"__z__{c}"
                    hv = safe_float(hr.get(zc))
                    lv = safe_float(lr.get(zc))
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
    out: dict[str, Any] = {
        "pairs": int(len(pairs)),
        "max_abs_smd": math.nan,
    }
    vals = []
    for c in controls:
        v = smd(pairs.get(f"high__{c}", pd.Series(dtype=float)),
                pairs.get(f"low__{c}", pd.Series(dtype=float)))
        out[f"smd__{c}"] = v
        if math.isfinite(v):
            vals.append(abs(v))
    out["max_abs_smd"] = max(vals) if vals else math.nan
    return out


def repair_balance(
    pairs: pd.DataFrame,
    *,
    controls: Sequence[str],
    min_pairs: int,
) -> pd.DataFrame:
    """
    Deterministic outcome-blind trimming.

    Remove the pair most contributing to the currently worst control imbalance
    until every |SMD| <= BALANCE_SMD_GATE, or until removing further pairs would
    violate the predeclared minimum-pair gate.
    """
    if pairs.empty:
        return pairs.copy()

    x = pairs.copy().reset_index(drop=True)

    while len(x) > min_pairs:
        b = pair_balance(x, controls)
        smds = {
            c: safe_float(b.get(f"smd__{c}"))
            for c in controls
        }
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

        # Remove the pair that contributes most in the direction of imbalance.
        # Stable tie break: pair IDs/candidate IDs.
        order = pd.DataFrame(
            {
                "idx": x.index,
                "contribution": contribution.fillna(-np.inf),
                "tie": (
                    x["high_candidate_id"].astype(str)
                    + "|"
                    + x["low_candidate_id"].astype(str)
                ),
            }
        ).sort_values(
            ["contribution", "tie"],
            ascending=[False, False],
            kind="mergesort",
        )

        drop_idx = int(order.iloc[0]["idx"])
        x = x.drop(index=drop_idx).reset_index(drop=True)

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
    """
    Build/freeze all outcome-blind pair manifests and feasibility.
    """
    pair_parts = []
    gate_rows = []
    pair_counter = 0

    configs = []
    # Base: all horizons for robustness.
    for h in EXPECTED_HORIZONS:
        for feature, _sign, _family in BOOK_FEATURES:
            configs.append(
                (h, feature, "BASE", "ALL", BASE_CONTROL_COLUMNS, MIN_FULL_PAIRS)
            )
    # Strict static only at primary horizon.
    for feature, _sign, _family in BOOK_FEATURES:
        configs.append(
            (
                PRIMARY_OBSERVATION_HORIZON_SEC,
                feature,
                "STRICT_STATIC",
                "ALL",
                STRICT_CONTROL_COLUMNS,
                MIN_FULL_PAIRS,
            )
        )
    # Reaccel diagnostic only primary horizon/base.
    if "is_reaccel_context" in causal.columns:
        for feature, _sign, _family in BOOK_FEATURES:
            configs.append(
                (
                    PRIMARY_OBSERVATION_HORIZON_SEC,
                    feature,
                    "BASE",
                    "REACCEL_CONTEXT",
                    BASE_CONTROL_COLUMNS,
                    MIN_REACCEL_PAIRS,
                )
            )

    for i, (h, feature, tier, population, controls, min_pairs) in enumerate(
        configs, 1
    ):
        d = causal[
            pd.to_numeric(
                causal["observation_horizon_sec"], errors="coerce"
            ) == int(h)
        ].copy()
        d = d[
            population_mask(d, population)
            & bool_series(d["window_valid"])
            & bool_series(d["decision_price_valid"])
        ].copy()

        bcol = f"__book_tercile__{feature}"
        need = [bcol, f"__oriented__{feature}"] + list(controls)
        for c in need:
            d = d[pd.to_numeric(d[c], errors="coerce").notna()].copy()

        d = add_z_controls(d, controls)
        for c in controls:
            d = d[pd.to_numeric(d[f"__z__{c}"], errors="coerce").notna()].copy()

        high = d[pd.to_numeric(d[bcol], errors="coerce") == BOOK_BINS].copy()
        low = d[pd.to_numeric(d[bcol], errors="coerce") == 1].copy()

        raw = greedy_match(
            high,
            low,
            feature=feature,
            control_tier=tier,
            controls=controls,
            population=population,
            observation_horizon_sec=int(h),
        )
        pre = pair_balance(raw, controls)
        repaired = repair_balance(
            raw,
            controls=controls,
            min_pairs=int(min_pairs),
        )
        post = pair_balance(repaired, controls)

        balance_pass = bool(
            len(repaired) > 0
            and math.isfinite(safe_float(post["max_abs_smd"]))
            and safe_float(post["max_abs_smd"]) <= BALANCE_SMD_GATE + 1e-12
        )

        disc_n = int(
            (
                repaired["candidate_date"].astype(str) <= DISCOVERY_END_DATE
            ).sum()
        ) if len(repaired) else 0
        val_n = int(
            (
                repaired["candidate_date"].astype(str) >= VALIDATION_START_DATE
            ).sum()
        ) if len(repaired) else 0

        total_gate = len(repaired) >= int(min_pairs) and balance_pass
        split_gate = (
            disc_n >= MIN_SPLIT_PAIRS
            and val_n >= MIN_SPLIT_PAIRS
            if population == "ALL"
            and h == PRIMARY_OBSERVATION_HORIZON_SEC
            else True
        )
        feasible = bool(total_gate and split_gate)

        if len(repaired):
            repaired = repaired.copy()
            repaired["pair_id"] = [
                f"7M4-{pair_counter + j + 1:09d}"
                for j in range(len(repaired))
            ]
            pair_counter += len(repaired)
            repaired["min_pairs_gate"] = int(min_pairs)
            repaired["final_balance_pass"] = balance_pass
            repaired["feasible_before_outcomes"] = feasible
            repaired["discovery_pairs"] = disc_n
            repaired["validation_pairs"] = val_n
            pair_parts.append(repaired)

        row = {
            "observation_horizon_sec": int(h),
            "feature": feature,
            "control_tier": tier,
            "population": population,
            "eligible_rows": int(len(d)),
            "high_tercile_rows": int(len(high)),
            "low_tercile_rows": int(len(low)),
            "pairs_before_repair": int(len(raw)),
            "pairs_after_repair": int(len(repaired)),
            "pairs_trimmed": int(len(raw) - len(repaired)),
            "min_pairs_gate": int(min_pairs),
            "discovery_pairs": disc_n,
            "validation_pairs": val_n,
            "min_split_pairs": (
                MIN_SPLIT_PAIRS
                if population == "ALL"
                and h == PRIMARY_OBSERVATION_HORIZON_SEC
                else 0
            ),
            "pre_max_abs_smd": safe_float(pre["max_abs_smd"]),
            "post_max_abs_smd": safe_float(post["max_abs_smd"]),
            "balance_smd_gate": BALANCE_SMD_GATE,
            "balance_pass": balance_pass,
            "split_gate_pass": split_gate,
            "feasible_before_outcomes": feasible,
        }
        for c in controls:
            row[f"pre_smd__{c}"] = safe_float(pre.get(f"smd__{c}"))
            row[f"post_smd__{c}"] = safe_float(post.get(f"smd__{c}"))
        gate_rows.append(row)

        if i == 1 or i % 8 == 0 or i == len(configs):
            print(
                f"      matching [{i:02d}/{len(configs):02d}] "
                f"h={h}s tier={tier:13s} pop={population:16s} "
                f"feature={feature[:40]} pairs={len(repaired):,} "
                f"max|SMD|={safe_float(post['max_abs_smd']):.3f} "
                f"feasible={feasible}",
                flush=True,
            )

    pairs = (
        pd.concat(pair_parts, ignore_index=True)
        if pair_parts
        else pd.DataFrame()
    )
    gate = pd.DataFrame(gate_rows)
    return pairs, gate


def load_outcomes_after_freeze(outcome_path: Path) -> pd.DataFrame:
    """
    Intentionally called only AFTER causal inputs/pairs/feasibility are written.
    """
    require_stack()
    outcomes = pq.read_table(outcome_path).to_pandas()
    require_columns(
        outcomes,
        [
            "candidate_id",
            "observation_horizon_sec",
            "decision_price_valid",
            "h300_fully_observed",
            "h600_fully_observed",
            "h900_fully_observed",
            "h1800_fully_observed",
            "h900_origin_reach_50",
            "h900_origin_reach_75",
            "h900_origin_reach_100",
        ],
        "Phase7M.3 candidate_decision_outcomes",
    )
    if outcomes.duplicated(
        ["candidate_id", "observation_horizon_sec"]
    ).any():
        raise RuntimeError("Outcome table duplicate candidate/horizon keys.")
    return outcomes


def available_targets(outcomes: pd.DataFrame, h: int) -> list[int]:
    out = []
    for target in (35, 50, 75, 100):
        if f"h{h}_origin_reach_{target}" in outcomes.columns:
            out.append(target)
    return out


def pair_outcome_frame(
    pairs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
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

    x = (
        pairs.merge(
            hi,
            left_on=["high_candidate_id", "observation_horizon_sec"],
            right_on=["high__candidate_id", "high__observation_horizon_sec"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            lo,
            left_on=["low_candidate_id", "observation_horizon_sec"],
            right_on=["low__candidate_id", "low__observation_horizon_sec"],
            how="left",
            validate="many_to_one",
        )
    )
    return x


def paired_effect(
    pairs: pd.DataFrame,
    *,
    target: int,
    outcome_horizon_sec: int,
) -> dict[str, Any]:
    if pairs.empty:
        return {
            "pairs": 0,
            "high_rate": math.nan,
            "low_rate": math.nan,
            "effect_pp": math.nan,
            "high_wins": 0,
            "low_wins": 0,
        }

    fully_h = f"high__h{outcome_horizon_sec}_fully_observed"
    fully_l = f"low__h{outcome_horizon_sec}_fully_observed"
    y_h = f"high__h{outcome_horizon_sec}_origin_reach_{target}"
    y_l = f"low__h{outcome_horizon_sec}_origin_reach_{target}"

    if any(c not in pairs.columns for c in (fully_h, fully_l, y_h, y_l)):
        return {
            "pairs": 0,
            "high_rate": math.nan,
            "low_rate": math.nan,
            "effect_pp": math.nan,
            "high_wins": 0,
            "low_wins": 0,
        }

    obs = (
        bool_series(pairs[fully_h])
        & bool_series(pairs[fully_l])
        & bool_series(pairs["high__decision_price_valid"])
        & bool_series(pairs["low__decision_price_valid"])
    )
    x = pairs.loc[obs].copy()

    hy = x[y_h].astype("boolean")
    ly = x[y_l].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid]
    hy = hy.loc[valid].astype(bool)
    ly = ly.loc[valid].astype(bool)

    if not len(x):
        return {
            "pairs": 0,
            "high_rate": math.nan,
            "low_rate": math.nan,
            "effect_pp": math.nan,
            "high_wins": 0,
            "low_wins": 0,
        }

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
    pairs: pd.DataFrame,
    *,
    target: int,
    outcome_horizon_sec: int,
    reps: int,
    seed: int,
) -> tuple[float, float, int]:
    fully_h = f"high__h{outcome_horizon_sec}_fully_observed"
    fully_l = f"low__h{outcome_horizon_sec}_fully_observed"
    y_h = f"high__h{outcome_horizon_sec}_origin_reach_{target}"
    y_l = f"low__h{outcome_horizon_sec}_origin_reach_{target}"

    if any(c not in pairs.columns for c in (fully_h, fully_l, y_h, y_l)):
        return math.nan, math.nan, 0

    obs = (
        bool_series(pairs[fully_h])
        & bool_series(pairs[fully_l])
        & bool_series(pairs["high__decision_price_valid"])
        & bool_series(pairs["low__decision_price_valid"])
    )
    x = pairs.loc[obs].copy()
    hy = x[y_h].astype("boolean")
    ly = x[y_l].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid].copy()
    if x.empty:
        return math.nan, math.nan, 0

    x["__diff"] = (
        x[y_h].astype(bool).astype(int)
        - x[y_l].astype(bool).astype(int)
    )
    g = (
        x.groupby(x["candidate_date"].astype(str))["__diff"]
        .agg(["sum", "count"])
    )
    if len(g) < 2:
        return math.nan, math.nan, 0

    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    n_days = len(g)

    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        n_days,
        np.full(n_days, 1.0 / n_days),
        size=int(reps),
    )
    den = multiplicity @ counts
    vals = np.divide(
        multiplicity @ sums,
        den,
        out=np.full(int(reps), np.nan),
        where=den > 0,
    ) * 100.0
    vals = vals[np.isfinite(vals)]

    if len(vals) < MIN_BOOTSTRAP_VALID_REPS:
        return math.nan, math.nan, int(len(vals))

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def primary_result_label(
    effect_pp: float,
    ci_low: float,
    feasible: bool,
) -> str:
    if not feasible:
        return "NOT_TESTED"
    if not math.isfinite(effect_pp):
        return "NOT_TESTED"
    if effect_pp <= 0:
        return "NOT_SUPPORTED"
    if math.isfinite(ci_low) and ci_low > 0:
        return "SUPPORTED"
    return "TENTATIVE"


def analyze_primary(
    joined_pairs: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate_map = {
        (
            int(r.observation_horizon_sec),
            str(r.feature),
            str(r.control_tier),
            str(r.population),
        ): bool(r.feasible_before_outcomes)
        for r in gate.itertuples(index=False)
    }

    primary_rows = []
    split_rows = []
    static_rows = []

    for feature, sign, family in BOOK_FEATURES:
        for tier in ("BASE", "STRICT_STATIC"):
            p = joined_pairs[
                (
                    pd.to_numeric(
                        joined_pairs["observation_horizon_sec"],
                        errors="coerce",
                    )
                    == PRIMARY_OBSERVATION_HORIZON_SEC
                )
                & (joined_pairs["feature"] == feature)
                & (joined_pairs["control_tier"] == tier)
                & (joined_pairs["population"] == "ALL")
            ].copy()

            feasible = gate_map.get(
                (
                    PRIMARY_OBSERVATION_HORIZON_SEC,
                    feature,
                    tier,
                    "ALL",
                ),
                False,
            )

            for target in available_targets_from_joined(
                p, PRIMARY_OUTCOME_HORIZON_SEC
            ):
                result = paired_effect(
                    p,
                    target=target,
                    outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                )
                if feasible:
                    lo, hi, nboot = bootstrap_matched_day(
                        p,
                        target=target,
                        outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                        reps=bootstrap_reps,
                        seed=(
                            BOOTSTRAP_SEED
                            + sum(ord(c) for c in feature)
                            + target * 100
                            + (50_000 if tier == "STRICT_STATIC" else 0)
                        ),
                    )
                else:
                    lo = hi = math.nan
                    nboot = 0

                row = {
                    "feature": feature,
                    "feature_family": family,
                    "feature_orientation_sign": sign,
                    "control_tier": tier,
                    "target_bps": int(target),
                    "observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
                    "outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
                    "feasible_before_outcomes": feasible,
                    **result,
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "bootstrap_valid_reps": nboot,
                    "interpretation": primary_result_label(
                        safe_float(result["effect_pp"]), lo, feasible
                    ),
                }
                primary_rows.append(row)
                if tier == "STRICT_STATIC":
                    static_rows.append(row)

            # Discovery / validation only primary P75.
            for split_name, split_mask in (
                (
                    "DISCOVERY",
                    p["candidate_date"].astype(str) <= DISCOVERY_END_DATE,
                ),
                (
                    "VALIDATION",
                    p["candidate_date"].astype(str) >= VALIDATION_START_DATE,
                ),
            ):
                sp = p.loc[split_mask].copy()
                split_feasible = bool(feasible and len(sp) >= MIN_SPLIT_PAIRS)
                result = paired_effect(
                    sp,
                    target=PRIMARY_TARGET_BPS,
                    outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                )
                if split_feasible:
                    lo, hi, nboot = bootstrap_matched_day(
                        sp,
                        target=PRIMARY_TARGET_BPS,
                        outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                        reps=bootstrap_reps,
                        seed=(
                            BOOTSTRAP_SEED
                            + sum(ord(c) for c in feature + tier + split_name)
                        ),
                    )
                else:
                    lo = hi = math.nan
                    nboot = 0
                split_rows.append(
                    {
                        "feature": feature,
                        "feature_family": family,
                        "control_tier": tier,
                        "split": split_name,
                        "pairs_available_before_outcome_filter": int(len(sp)),
                        "min_split_pairs": MIN_SPLIT_PAIRS,
                        "split_feasible": split_feasible,
                        **result,
                        "ci_low_pp": lo,
                        "ci_high_pp": hi,
                        "bootstrap_valid_reps": nboot,
                        "direction_positive": bool(
                            math.isfinite(safe_float(result["effect_pp"]))
                            and safe_float(result["effect_pp"]) > 0
                        ),
                    }
                )

    primary = pd.DataFrame(primary_rows)
    split = pd.DataFrame(split_rows)
    static = pd.DataFrame(static_rows)

    # Add chronological replication flag to primary P75 rows.
    if len(primary) and len(split):
        flags = (
            split.groupby(["feature", "control_tier"])
            .agg(
                discovery_positive=(
                    "direction_positive",
                    lambda x: bool(x.iloc[0]) if len(x) else False,
                ),
            )
        )
        # Easier explicit map, because split order is deterministic but group agg
        # should not encode split name semantics.
        rep = {}
        for (feature, tier), g in split.groupby(["feature", "control_tier"]):
            d = g[g["split"] == "DISCOVERY"]
            v = g[g["split"] == "VALIDATION"]
            rep[(feature, tier)] = bool(
                len(d)
                and len(v)
                and bool(d.iloc[0]["split_feasible"])
                and bool(v.iloc[0]["split_feasible"])
                and safe_float(d.iloc[0]["effect_pp"]) > 0
                and safe_float(v.iloc[0]["effect_pp"]) > 0
            )
        primary["chronological_direction_replication"] = [
            rep.get((str(r.feature), str(r.control_tier)), False)
            if int(r.target_bps) == PRIMARY_TARGET_BPS
            else False
            for r in primary.itertuples(index=False)
        ]

    return primary, split, static


def available_targets_from_joined(p: pd.DataFrame, h: int) -> list[int]:
    targets = []
    for t in (35, 50, 75, 100):
        if (
            f"high__h{h}_origin_reach_{t}" in p.columns
            and f"low__h{h}_origin_reach_{t}" in p.columns
        ):
            targets.append(t)
    return targets


def analyze_observation_horizon_robustness(
    joined_pairs: pd.DataFrame,
    gate: pd.DataFrame,
) -> pd.DataFrame:
    gate_map = {
        (
            int(r.observation_horizon_sec),
            str(r.feature),
            str(r.control_tier),
            str(r.population),
        ): bool(r.feasible_before_outcomes)
        for r in gate.itertuples(index=False)
    }
    rows = []
    for h in EXPECTED_HORIZONS:
        for feature, sign, family in BOOK_FEATURES:
            p = joined_pairs[
                (
                    pd.to_numeric(
                        joined_pairs["observation_horizon_sec"],
                        errors="coerce",
                    )
                    == h
                )
                & (joined_pairs["feature"] == feature)
                & (joined_pairs["control_tier"] == "BASE")
                & (joined_pairs["population"] == "ALL")
            ].copy()
            feasible = gate_map.get((h, feature, "BASE", "ALL"), False)
            result = paired_effect(
                p,
                target=PRIMARY_TARGET_BPS,
                outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_family": family,
                    "feature_orientation_sign": sign,
                    "observation_horizon_sec": h,
                    "target_bps": PRIMARY_TARGET_BPS,
                    "outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
                    "feasible_before_outcomes": feasible,
                    **result,
                    "role": (
                        "PRIMARY"
                        if h == PRIMARY_OBSERVATION_HORIZON_SEC
                        else "ROBUSTNESS_ONLY"
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze_outcome_horizon_robustness(
    joined_pairs: pd.DataFrame,
    gate: pd.DataFrame,
) -> pd.DataFrame:
    gate_map = {
        (str(r.feature), str(r.control_tier)): bool(r.feasible_before_outcomes)
        for r in gate.itertuples(index=False)
        if int(r.observation_horizon_sec) == PRIMARY_OBSERVATION_HORIZON_SEC
        and str(r.population) == "ALL"
    }
    rows = []
    for feature, sign, family in BOOK_FEATURES:
        for tier in ("BASE", "STRICT_STATIC"):
            p = joined_pairs[
                (
                    pd.to_numeric(
                        joined_pairs["observation_horizon_sec"],
                        errors="coerce",
                    )
                    == PRIMARY_OBSERVATION_HORIZON_SEC
                )
                & (joined_pairs["feature"] == feature)
                & (joined_pairs["control_tier"] == tier)
                & (joined_pairs["population"] == "ALL")
            ].copy()
            feasible = gate_map.get((feature, tier), False)
            for oh in OUTCOME_HORIZONS_SEC:
                result = paired_effect(
                    p,
                    target=PRIMARY_TARGET_BPS,
                    outcome_horizon_sec=oh,
                )
                rows.append(
                    {
                        "feature": feature,
                        "feature_family": family,
                        "feature_orientation_sign": sign,
                        "control_tier": tier,
                        "observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
                        "outcome_horizon_sec": oh,
                        "target_bps": PRIMARY_TARGET_BPS,
                        "feasible_before_outcomes": feasible,
                        **result,
                        "role": (
                            "PRIMARY"
                            if oh == PRIMARY_OUTCOME_HORIZON_SEC
                            else "ROBUSTNESS_ONLY"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def analyze_direction_robustness(
    joined_pairs: pd.DataFrame,
    gate: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature, sign, family in BOOK_FEATURES:
        for tier in ("BASE", "STRICT_STATIC"):
            p = joined_pairs[
                (
                    pd.to_numeric(
                        joined_pairs["observation_horizon_sec"],
                        errors="coerce",
                    )
                    == PRIMARY_OBSERVATION_HORIZON_SEC
                )
                & (joined_pairs["feature"] == feature)
                & (joined_pairs["control_tier"] == tier)
                & (joined_pairs["population"] == "ALL")
            ].copy()
            for direction, label in ((1, "UP"), (-1, "DOWN")):
                sp = p[
                    pd.to_numeric(p["direction"], errors="coerce") == direction
                ].copy()
                result = paired_effect(
                    sp,
                    target=PRIMARY_TARGET_BPS,
                    outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                )
                rows.append(
                    {
                        "feature": feature,
                        "feature_family": family,
                        "control_tier": tier,
                        "direction": direction,
                        "direction_label": label,
                        "pairs_available_before_outcome_filter": int(len(sp)),
                        "direction_pair_gate": MIN_DIRECTION_PAIRS,
                        "direction_diagnostic_feasible": bool(
                            len(sp) >= MIN_DIRECTION_PAIRS
                        ),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def analyze_reacceleration(
    joined_pairs: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    bootstrap_reps: int,
) -> pd.DataFrame:
    if not (joined_pairs["population"] == "REACCEL_CONTEXT").any():
        return pd.DataFrame()

    gate_map = {
        str(r.feature): bool(r.feasible_before_outcomes)
        for r in gate.itertuples(index=False)
        if int(r.observation_horizon_sec) == PRIMARY_OBSERVATION_HORIZON_SEC
        and str(r.control_tier) == "BASE"
        and str(r.population) == "REACCEL_CONTEXT"
    }

    rows = []
    for feature, sign, family in BOOK_FEATURES:
        p = joined_pairs[
            (
                pd.to_numeric(
                    joined_pairs["observation_horizon_sec"],
                    errors="coerce",
                )
                == PRIMARY_OBSERVATION_HORIZON_SEC
            )
            & (joined_pairs["feature"] == feature)
            & (joined_pairs["control_tier"] == "BASE")
            & (joined_pairs["population"] == "REACCEL_CONTEXT")
        ].copy()
        feasible = gate_map.get(feature, False)
        for target in available_targets_from_joined(
            p, PRIMARY_OUTCOME_HORIZON_SEC
        ):
            result = paired_effect(
                p,
                target=target,
                outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
            )
            if feasible:
                lo, hi, nboot = bootstrap_matched_day(
                    p,
                    target=target,
                    outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                    reps=bootstrap_reps,
                    seed=(
                        BOOTSTRAP_SEED
                        + 700_000
                        + sum(ord(c) for c in feature)
                        + target * 10
                    ),
                )
            else:
                lo = hi = math.nan
                nboot = 0
            rows.append(
                {
                    "feature": feature,
                    "feature_family": family,
                    "target_bps": target,
                    "feasible_before_outcomes": feasible,
                    **result,
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "bootstrap_valid_reps": nboot,
                    "interpretation": primary_result_label(
                        safe_float(result["effect_pp"]), lo, feasible
                    ),
                }
            )
    return pd.DataFrame(rows)


def day_sign_consistency(joined_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, _sign, family in BOOK_FEATURES:
        for tier in ("BASE", "STRICT_STATIC"):
            p = joined_pairs[
                (
                    pd.to_numeric(
                        joined_pairs["observation_horizon_sec"],
                        errors="coerce",
                    )
                    == PRIMARY_OBSERVATION_HORIZON_SEC
                )
                & (joined_pairs["feature"] == feature)
                & (joined_pairs["control_tier"] == tier)
                & (joined_pairs["population"] == "ALL")
            ].copy()
            effects = []
            for day, g in p.groupby(p["candidate_date"].astype(str)):
                r = paired_effect(
                    g,
                    target=PRIMARY_TARGET_BPS,
                    outcome_horizon_sec=PRIMARY_OUTCOME_HORIZON_SEC,
                )
                if int(r["pairs"]) >= 3 and math.isfinite(safe_float(r["effect_pp"])):
                    effects.append((str(day), safe_float(r["effect_pp"])))
            vals = [x[1] for x in effects]
            rows.append(
                {
                    "feature": feature,
                    "feature_family": family,
                    "control_tier": tier,
                    "comparable_days": len(vals),
                    "positive_days": sum(v > 0 for v in vals),
                    "negative_days": sum(v < 0 for v in vals),
                    "zero_days": sum(v == 0 for v in vals),
                    "positive_day_share": (
                        sum(v > 0 for v in vals) / len(vals)
                        if vals
                        else math.nan
                    ),
                    "median_day_effect_pp": (
                        float(np.median(vals)) if vals else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_invariants(
    *,
    causal: pd.DataFrame,
    pairs: pd.DataFrame,
    gate: pd.DataFrame,
    outcomes: pd.DataFrame,
    static_book: pd.DataFrame,
    m3_meta: dict[str, Any],
    bridge: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append(
            {
                "invariant": name,
                "status": "PASS" if bool(ok) else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    candidate_n = causal["candidate_id"].astype(str).nunique()
    add(
        "expected_candidate_population",
        candidate_n == EXPECTED_CANDIDATES,
        candidate_n,
        EXPECTED_CANDIDATES,
    )

    horizons = tuple(
        sorted(
            pd.to_numeric(
                causal["observation_horizon_sec"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )
    )
    add(
        "expected_horizons",
        horizons == EXPECTED_HORIZONS,
        horizons,
        EXPECTED_HORIZONS,
    )
    add(
        "causal_row_count",
        len(causal) == EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS),
        len(causal),
        EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS),
    )

    future_leaks = [
        c
        for c in causal.columns
        if (
            c.startswith("h300_")
            or c.startswith("h600_")
            or c.startswith("h900_")
            or c.startswith("h1800_")
            or "origin_reach_" in c.lower()
        )
    ]
    add(
        "causal_analysis_inputs_future_clean",
        len(future_leaks) == 0,
        future_leaks,
        [],
    )

    pair_leaks = [
        c
        for c in pairs.columns
        if (
            "origin_reach_" in c.lower()
            or "fully_observed" in c.lower()
            or "outcome" in c.lower()
        )
    ]
    add(
        "matched_pair_manifest_future_clean",
        len(pair_leaks) == 0,
        pair_leaks,
        [],
    )

    feasible_bad = gate[
        gate["feasible_before_outcomes"].astype(bool)
        & (
            pd.to_numeric(gate["post_max_abs_smd"], errors="coerce")
            > BALANCE_SMD_GATE + 1e-12
        )
    ]
    add(
        "all_feasible_matches_balanced",
        len(feasible_bad) == 0,
        len(feasible_bad),
        0,
    )

    exact_static_cov = float(
        (
            pd.to_numeric(
                static_book["initial_support_quote_5bps_exact"],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                static_book["initial_opposing_quote_5bps_exact"],
                errors="coerce",
            ).notna()
        ).mean()
    )
    add(
        "exact_static_book_coverage_at_least_95pct",
        exact_static_cov >= 0.95,
        exact_static_cov,
        ">=0.95",
    )

    p3 = causal[
        pd.to_numeric(
            causal["observation_horizon_sec"], errors="coerce"
        ) == PRIMARY_OBSERVATION_HORIZON_SEC
    ]
    exact_cov = float(bool_series(p3["window_valid"]).mean())
    add(
        "primary_3s_exact_mechanics_coverage_at_least_95pct",
        exact_cov >= 0.95,
        exact_cov,
        ">=0.95",
    )

    add(
        "phase7m3_metadata_candidate_count",
        int(m3_meta.get("candidate_count", -1)) == EXPECTED_CANDIDATES,
        m3_meta.get("candidate_count"),
        EXPECTED_CANDIDATES,
    )

    add(
        "outcome_keys_unique",
        not outcomes.duplicated(
            ["candidate_id", "observation_horizon_sec"]
        ).any(),
        int(
            outcomes.duplicated(
                ["candidate_id", "observation_horizon_sec"]
            ).sum()
        ),
        0,
    )

    add(
        "mechanism_bridge_nonempty",
        len(bridge) == len(BOOK_FEATURES),
        len(bridge),
        len(BOOK_FEATURES),
    )

    primary_gate = gate[
        (
            pd.to_numeric(
                gate["observation_horizon_sec"], errors="coerce"
            )
            == PRIMARY_OBSERVATION_HORIZON_SEC
        )
        & (gate["control_tier"] == "BASE")
        & (gate["population"] == "ALL")
    ]
    add(
        "all_primary_base_features_have_gate_rows",
        len(primary_gate) == len(BOOK_FEATURES),
        len(primary_gate),
        len(BOOK_FEATURES),
    )

    return pd.DataFrame(rows)


def findings_text(
    *,
    status: str,
    bridge: pd.DataFrame,
    gate: pd.DataFrame,
    primary: pd.DataFrame,
    split: pd.DataFrame,
    static: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7M.4 — Prospective 25-bp exact mechanism attribution",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{status}**",
        "- Raw depth/trade replay: **NO**",
        "- Primary observation horizon: **3 seconds**",
        "- Primary outcome: **75 bps within 15 minutes from +3s decision**",
        "- Composite score: **NONE**",
        "",
        "## Outcome-blind mechanism bridge",
        "",
    ]

    for _, r in bridge.sort_values(
        "cheap_withdrawal_to_oriented_exact_spearman",
        ascending=False,
    ).iterrows():
        lines.append(
            f"- {r['feature']}: Spearman(cheap withdrawal, oriented exact)="
            f"{safe_float(r['cheap_withdrawal_to_oriented_exact_spearman']):+.3f}; "
            f"high-vs-low withdrawal robust standardized exact difference="
            f"{safe_float(r['robust_standardized_high_minus_low']):+.3f}; "
            f"N={int(r['paired_rows'])}."
        )

    lines += ["", "## Pre-outcome primary feasibility", ""]
    g = gate[
        (
            pd.to_numeric(
                gate["observation_horizon_sec"], errors="coerce"
            )
            == PRIMARY_OBSERVATION_HORIZON_SEC
        )
        & (gate["population"] == "ALL")
    ]
    for _, r in g.iterrows():
        lines.append(
            f"- {r['feature']} / {r['control_tier']}: "
            f"pairs={int(r['pairs_after_repair'])}; "
            f"max|SMD|={safe_float(r['post_max_abs_smd']):.3f}; "
            f"discovery={int(r['discovery_pairs'])}; "
            f"validation={int(r['validation_pairs'])}; "
            f"feasible={bool(r['feasible_before_outcomes'])}."
        )

    lines += ["", "## Primary 75-bp / 15-minute results", ""]
    p75 = primary[
        (primary["target_bps"] == PRIMARY_TARGET_BPS)
        & (
            primary["observation_horizon_sec"]
            == PRIMARY_OBSERVATION_HORIZON_SEC
        )
    ]
    for _, r in p75.iterrows():
        lines.append(
            f"- {r['feature']} / {r['control_tier']}: "
            f"high={100*safe_float(r['high_rate']):.1f}% vs "
            f"low={100*safe_float(r['low_rate']):.1f}%; "
            f"effect={safe_float(r['effect_pp']):+.2f}pp; "
            f"CI=[{safe_float(r['ci_low_pp']):+.2f}, "
            f"{safe_float(r['ci_high_pp']):+.2f}]pp; "
            f"N={int(r['pairs'])}; "
            f"{r['interpretation']}; "
            f"chronological-direction-replication="
            f"{bool(r.get('chronological_direction_replication', False))}."
        )

    lines += [
        "",
        "## Interpretation discipline",
        "",
        "- BASE controls momentum + executed directional aggression.",
        "- STRICT_STATIC additionally controls exact starting opposing/support frozen 5-bp quote.",
        "- A BASE effect that disappears under STRICT_STATIC should be interpreted as potentially starting-depth-driven.",
        "- A dynamic failure effect that survives STRICT_STATIC is stronger evidence for liquidity durability/failure rather than static thinness.",
        "- Retained opposing liquidity and durable unrecovered opposing loss are closely related queue-survival descriptions, not independent mechanisms.",
        "- Discovery and validation direction are reported separately; pooled significance alone is not called chronological replication.",
        "- No composite score, ML model, threshold optimization, or profitability claim is produced.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> int:
    # Basic deterministic matcher / balance / bootstrap / bridge tests.
    rows = []
    for i in range(360):
        direction = 1 if i % 2 == 0 else -1
        day = f"2026-06-{1 + (i % 12):02d}"
        failure = i / 359.0
        rows.append(
            {
                "candidate_id": f"c{i:04d}",
                "candidate_date": day,
                "direction": direction,
                "direction_label": "UP" if direction > 0 else "DOWN",
                "observation_horizon_sec": 3,
                "window_valid": True,
                "decision_price_valid": True,
                PRIMARY_MOMENTUM_CONTROL: math.sin(i / 17.0),
                PRIMARY_AGGRESSION_CONTROL: 10_000 + (i % 30) * 100,
                CHEAP_WITHDRAWAL_FEATURE: failure,
                "proxy_valid": True,
                "initial_support_quote_5bps_exact": 1_000_000 + (i % 20) * 1000,
                "initial_opposing_quote_5bps_exact": 900_000 + (i % 20) * 1000,
                "is_reaccel_context": i % 3 == 0,
                "opposing__durable_unrecovered_loss_share_initial": failure,
                "opposing__peak_initial_queue_loss_share": failure,
                "opposing__retained_initial_quote_ratio": 1.0 - failure,
                "support_minus_opposing_retained_ratio": failure,
                "opposing__same_price_refill_to_depletion_ratio": 1.0 - failure,
                "opposing__durable_same_price_refill_fraction": 1.0 - failure,
                "support__retained_initial_quote_ratio": failure,
                "support__same_price_refill_to_depletion_ratio": failure,
                "__control_momentum": math.sin(i / 17.0),
                "__control_aggression": math.log1p(10_000 + (i % 30) * 100),
                "__control_initial_opposing": math.log1p(
                    900_000 + (i % 20) * 1000
                ),
                "__control_initial_support": math.log1p(
                    1_000_000 + (i % 20) * 1000
                ),
            }
        )
    d = pd.DataFrame(rows)

    for feature, sign, _ in BOOK_FEATURES:
        d[f"__oriented__{feature}"] = (
            pd.to_numeric(d[feature], errors="coerce") * sign
        )
        d[f"__book_tercile__{feature}"] = percentile_bin(
            d[f"__oriented__{feature}"], 3
        ).astype(float)

    bridge, bridge_t = mechanism_bridge(d)
    assert len(bridge) == len(BOOK_FEATURES)
    assert (
        bridge["cheap_withdrawal_to_oriented_exact_spearman"]
        .dropna()
        .min()
        > 0.9
    )

    feature = BOOK_FEATURES[0][0]
    x = add_z_controls(d, BASE_CONTROL_COLUMNS)
    hi = x[x[f"__book_tercile__{feature}"] == 3]
    lo = x[x[f"__book_tercile__{feature}"] == 1]
    pairs = greedy_match(
        hi,
        lo,
        feature=feature,
        control_tier="BASE",
        controls=BASE_CONTROL_COLUMNS,
        population="ALL",
        observation_horizon_sec=3,
    )
    assert len(pairs) > 0
    repaired = repair_balance(
        pairs,
        controls=BASE_CONTROL_COLUMNS,
        min_pairs=min(50, len(pairs)),
    )
    bal = pair_balance(repaired, BASE_CONTROL_COLUMNS)
    assert len(repaired) > 0
    assert math.isfinite(safe_float(bal["max_abs_smd"]))

    # Matched day bootstrap synthetic outcome.
    po = repaired.copy()
    po["high__decision_price_valid"] = True
    po["low__decision_price_valid"] = True
    po["high__h900_fully_observed"] = True
    po["low__h900_fully_observed"] = True
    po["high__h900_origin_reach_75"] = [
        (i % 3 != 0) for i in range(len(po))
    ]
    po["low__h900_origin_reach_75"] = [
        (i % 3 == 0) for i in range(len(po))
    ]
    eff = paired_effect(po, target=75, outcome_horizon_sec=900)
    assert eff["pairs"] == len(po)
    lo_ci, hi_ci, reps = bootstrap_matched_day(
        po,
        target=75,
        outcome_horizon_sec=900,
        reps=300,
        seed=1,
    )
    assert reps >= MIN_BOOTSTRAP_VALID_REPS
    assert math.isfinite(lo_ci) and math.isfinite(hi_ci)

    # Outcome join contract:
    # one candidate/horizon can legitimately appear in multiple frozen pair
    # sets (different features / control tiers / populations). The outcome
    # table remains unique per candidate/horizon, so the correct join is
    # many-to-one from pair rows to outcome rows.
    repeated_pairs = pd.DataFrame(
        {
            "feature": ["f1", "f2"],
            "control_tier": ["BASE", "STRICT_STATIC"],
            "population": ["ALL", "ALL"],
            "observation_horizon_sec": [3, 3],
            "candidate_date": ["2026-06-01", "2026-06-01"],
            "direction": [1, 1],
            "high_candidate_id": ["h1", "h1"],
            "low_candidate_id": ["l1", "l1"],
        }
    )
    unique_outcomes = pd.DataFrame(
        {
            "candidate_id": ["h1", "l1"],
            "observation_horizon_sec": [3, 3],
            "decision_price_valid": [True, True],
            "h300_fully_observed": [True, True],
            "h600_fully_observed": [True, True],
            "h900_fully_observed": [True, True],
            "h1800_fully_observed": [True, True],
            "h900_origin_reach_50": [True, False],
            "h900_origin_reach_75": [True, False],
            "h900_origin_reach_100": [False, False],
        }
    )
    joined = pair_outcome_frame(repeated_pairs, unique_outcomes)
    assert len(joined) == len(repeated_pairs)
    assert joined["high__candidate_id"].notna().all()
    assert joined["low__candidate_id"].notna().all()

    print("SELF-TEST PASS")
    print(
        "Validated outcome-blind mechanism bridge, deterministic exact-block "
        "matching, SMD balance repair, candidate-day matched bootstrap, and "
        "many-to-one outcome joins across repeated feature/tier pair sets."
    )
    return 0


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Phase 7M.4 prospective 25-bp exact mechanism attribution"
    )
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--m2-root", type=Path, default=DEFAULT_M2_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    require_stack()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    print("Building Phase 7M.4 prospective 25-bp exact mechanism attribution...")
    print(f"Builder version         : {BUILDER_VERSION}")
    print("Raw replay              : NONE")
    print("Primary observation     : 3s")
    print("Primary outcome         : origin reaches 75bp within 15m")
    print("Base controls           : exact date+direction + momentum + aggression")
    print("Strict controls         : BASE + exact initial support/opposing frozen 5bp quote")
    print("Chronological split     : discovery<=2026-06-17 / validation>=2026-06-18")
    print("Composite / ML          : NONE")

    print("[1/9] Loading frozen Phase 7M.3 / 7M.2 causal contracts...")
    (
        m3_meta,
        m2_meta,
        candidate_inputs,
        mechanics,
        cheap_inputs,
        outcome_path,
    ) = load_frozen_inputs(args.m3_root, args.m2_root)

    candidate_ids = set(candidate_inputs["candidate_id"].astype(str))
    print(
        f"      candidates={len(candidate_ids):,}; mechanics={len(mechanics):,}; "
        f"3s coverage="
        f"{100*bool_series(mechanics[pd.to_numeric(mechanics['observation_horizon_sec'],errors='coerce')==3]['window_valid']).mean():.3f}%"
    )

    print("[2/9] Deriving exact starting-book controls from frozen level audit...")
    static_book, static_summary = load_exact_initial_book_state(
        args.m3_root, candidate_ids
    )
    print(
        f"      exact static coverage="
        f"{100*safe_float(static_summary['exact_static_coverage']):.3f}% "
        f"({static_summary['candidates_with_both_exact_sides']:,}/{EXPECTED_CANDIDATES:,})"
    )
    atomic_parquet(
        static_book,
        args.output_root / "exact_initial_book_state.parquet",
    )

    print("[3/9] Building OUTCOME-BLIND causal analysis table...")
    causal = build_causal_analysis_inputs(
        candidate_inputs,
        mechanics,
        cheap_inputs,
        static_book,
    )
    atomic_parquet(
        causal,
        args.output_root / "analysis_inputs.parquet",
    )

    print("[4/9] Building OUTCOME-BLIND cheap-withdrawal -> exact mechanism bridge...")
    bridge, bridge_terciles = mechanism_bridge(causal)
    atomic_csv(bridge, args.report_root / "mechanism_bridge.csv")
    atomic_csv(
        bridge_terciles,
        args.report_root / "mechanism_bridge_by_withdrawal_tercile.csv",
    )

    for _, r in bridge.iterrows():
        print(
            f"      bridge {r['feature'][:43]:43s} "
            f"rho={safe_float(r['cheap_withdrawal_to_oriented_exact_spearman']):+.3f} "
            f"RSD={safe_float(r['robust_standardized_high_minus_low']):+.3f}"
        )

    print("[5/9] PRE-COMPUTE matching / balance / feasibility BEFORE outcomes...")
    pairs, gate = build_pair_sets(causal)

    # Freeze causal manifests before outcomes are even loaded.
    atomic_parquet(
        pairs,
        args.output_root / "matched_pairs.parquet",
    )
    atomic_csv(
        gate,
        args.report_root / "precompute_feasibility_gate.csv",
    )
    atomic_csv(
        gate,
        args.report_root / "matching_balance.csv",
    )

    primary_gate = gate[
        (
            pd.to_numeric(
                gate["observation_horizon_sec"], errors="coerce"
            )
            == PRIMARY_OBSERVATION_HORIZON_SEC
        )
        & (gate["control_tier"].isin(["BASE", "STRICT_STATIC"]))
        & (gate["population"] == "ALL")
    ]
    for _, r in primary_gate.iterrows():
        print(
            f"      {r['control_tier']:13s} "
            f"{r['feature'][:42]:42s} "
            f"pairs={int(r['pairs_after_repair']):4d} "
            f"SMD={safe_float(r['post_max_abs_smd']):.3f} "
            f"D/V={int(r['discovery_pairs'])}/{int(r['validation_pairs'])} "
            f"FEASIBLE={bool(r['feasible_before_outcomes'])}"
        )

    print("[6/9] Causal manifests frozen. Loading future decision outcomes NOW...")
    outcomes = load_outcomes_after_freeze(outcome_path)
    joined_pairs = pair_outcome_frame(pairs, outcomes)
    if len(joined_pairs) != len(pairs):
        raise RuntimeError(
            "Outcome join changed frozen pair-manifest row count: "
            f"pairs={len(pairs):,}, joined={len(joined_pairs):,}"
        )
    print(
        f"      outcome join rows={len(joined_pairs):,}; "
        "pair manifest preserved exactly"
    )

    print("[7/9] Running predeclared prospective exact-mechanism tests...")
    primary, split, static = analyze_primary(
        joined_pairs,
        gate,
        bootstrap_reps=int(args.bootstrap_reps),
    )
    obs_robust = analyze_observation_horizon_robustness(joined_pairs, gate)
    out_robust = analyze_outcome_horizon_robustness(joined_pairs, gate)
    direction = analyze_direction_robustness(joined_pairs, gate)
    reaccel = analyze_reacceleration(
        joined_pairs,
        gate,
        bootstrap_reps=int(args.bootstrap_reps),
    )
    day_sign = day_sign_consistency(joined_pairs)

    atomic_csv(
        primary,
        args.report_root / "primary_exact_mechanism_contrasts.csv",
    )
    atomic_csv(
        split,
        args.report_root / "discovery_validation.csv",
    )
    atomic_csv(
        static,
        args.report_root / "static_state_robustness.csv",
    )
    atomic_csv(
        obs_robust,
        args.report_root / "observation_horizon_robustness.csv",
    )
    atomic_csv(
        out_robust,
        args.report_root / "outcome_horizon_robustness.csv",
    )
    atomic_csv(
        direction,
        args.report_root / "direction_robustness.csv",
    )
    atomic_csv(
        reaccel,
        args.report_root / "reacceleration_diagnostic.csv",
    )
    atomic_csv(
        day_sign,
        args.report_root / "feature_day_sign_consistency.csv",
    )

    print("[8/9] Checking invariants / data separation...")
    invariants = build_invariants(
        causal=causal,
        pairs=pairs,
        gate=gate,
        outcomes=outcomes,
        static_book=static_book,
        m3_meta=m3_meta,
        bridge=bridge,
    )
    atomic_csv(
        invariants,
        args.report_root / "analysis_invariants.csv",
    )
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    for _, r in invariants.iterrows():
        print(
            f"      {r['status']:4s} {r['invariant']}: "
            f"observed={r['observed']} expected={r['expected']}"
        )

    print("[9/9] Writing metadata / findings...")
    primary_p75 = primary[
        (primary["target_bps"] == PRIMARY_TARGET_BPS)
        & (primary["control_tier"] == "BASE")
    ].copy()

    supported_base = int(
        (primary_p75["interpretation"] == "SUPPORTED").sum()
    ) if len(primary_p75) else 0
    replicated_base = int(
        primary_p75.get(
            "chronological_direction_replication",
            pd.Series(dtype=bool),
        ).fillna(False).astype(bool).sum()
    ) if len(primary_p75) else 0

    strict_p75 = primary[
        (primary["target_bps"] == PRIMARY_TARGET_BPS)
        & (primary["control_tier"] == "STRICT_STATIC")
    ].copy()
    supported_strict = int(
        (strict_p75["interpretation"] == "SUPPORTED").sum()
    ) if len(strict_p75) else 0

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "built_at_utc": utc_now_iso(),
        "frozen_phase7m3_builder": FROZEN_PHASE7M3_BUILDER,
        "frozen_phase7m3_schema": FROZEN_PHASE7M3_SCHEMA,
        "frozen_phase7m2_builder": FROZEN_PHASE7M2_BUILDER,
        "candidate_count": EXPECTED_CANDIDATES,
        "candidate_id_set_sha256": sha256_ids(candidate_ids),
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_target_bps": PRIMARY_TARGET_BPS,
        "book_features_tested": [x[0] for x in BOOK_FEATURES],
        "core_book_features": list(CORE_BOOK_FEATURES),
        "base_control_columns": list(BASE_CONTROL_COLUMNS),
        "strict_static_control_columns": list(STRICT_CONTROL_COLUMNS),
        "min_full_pairs": MIN_FULL_PAIRS,
        "min_split_pairs": MIN_SPLIT_PAIRS,
        "min_reaccel_pairs": MIN_REACCEL_PAIRS,
        "balance_smd_gate": BALANCE_SMD_GATE,
        "supported_base_primary_75_features": supported_base,
        "chronologically_positive_base_primary_75_features": replicated_base,
        "supported_strict_static_primary_75_features": supported_strict,
        "failed_invariants": int(len(failed)),
        "raw_depth_replay": False,
        "raw_trade_replay": False,
        "composite_score": False,
        "ml": False,
        "scientific_state": "PROSPECTIVE_25BP_EXACT_MECHANISM_ATTRIBUTION_COMPLETE_NOT_ALPHA",
    }
    run_config = {
        **summary,
        "m3_root": str(args.m3_root),
        "m2_root": str(args.m2_root),
        "output_root": str(args.output_root),
        "report_root": str(args.report_root),
        "cheap_withdrawal_feature": CHEAP_WITHDRAWAL_FEATURE,
        "feature_contract": [
            {
                "feature": feature,
                "orientation_sign": sign,
                "family": family,
            }
            for feature, sign, family in BOOK_FEATURES
        ],
        "matching": {
            "exact_blocks": ["candidate_date", "direction"],
            "z_caliper": MATCH_Z_CALIPER,
            "euclidean_caliper": MATCH_EUCLIDEAN_CALIPER,
            "replacement": False,
            "balance_repair": "OUTCOME_BLIND_DETERMINISTIC_PAIR_TRIM",
            "balance_smd_gate": BALANCE_SMD_GATE,
        },
        "exact_static_state": {
            **static_summary,
            "definition": (
                "sum(price * initial_qty) across frozen Phase7M.3 level_audit "
                "levels, separately SUPPORT and OPPOSING"
            ),
        },
        "discovery_end_date": DISCOVERY_END_DATE,
        "validation_start_date": VALIDATION_START_DATE,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "future_labels_loaded_only_after_causal_pair_manifest_written": True,
    }

    atomic_json(
        args.output_root / "metadata" / "latest_summary.json",
        summary,
    )
    atomic_json(
        args.output_root / "metadata" / "run_config.json",
        run_config,
    )
    atomic_text(
        args.report_root / "findings.md",
        findings_text(
            status=status,
            bridge=bridge,
            gate=gate,
            primary=primary,
            split=split,
            static=static,
        ),
    )

    print("")
    print(f"Phase 7M.4 status: {status}")
    print(
        f"Supported BASE exact features for P75/15m: {supported_base}/"
        f"{len(BOOK_FEATURES)}"
    )
    print(
        f"Chronologically positive BASE P75 features: {replicated_base}/"
        f"{len(BOOK_FEATURES)}"
    )
    print(
        f"Supported STRICT_STATIC P75 features: {supported_strict}/"
        f"{len(BOOK_FEATURES)}"
    )
    print(f"Reports: {args.report_root}")
    print(
        "Interpret mechanism families, dependence, validation, and strict-static "
        "robustness together; do not rank features by p-value alone."
    )

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
