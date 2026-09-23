#!/usr/bin/env python3
"""
Phase 7L.4.1 — Outcome-blind matching balance repair + frozen re-acceleration test
=================================================================================

PLAIN-ENGLISH PROBLEM
---------------------
Phase 7L.4 found an encouraging re-acceleration pattern, but the primary matched
samples still had too much residual momentum/aggression imbalance.

So before trusting the 75/100 continuation difference, repair the matching
population WITHOUT LOOKING AT FUTURE OUTCOMES.

INPUT PAIRS
-----------
Reuse Phase 7L.4 matched_pairs.parquet.

Those pairs were already built:
* inside the strict causal re-acceleration context;
* same candidate date;
* same direction;
* high vs low tercile of ONE book mechanism;
* matched on continuous 3-second momentum and log aggressive notional;
* before future outcomes were joined.

CANDIDATE PAIR UNIVERSES
------------------------
For each individual book mechanism, Phase 7L.4 already produced:

    TIGHT
    PRIMARY
    RELAXED

caliper pair universes.

For each feature x caliper, this phase performs an OUTCOME-BLIND balance repair:

1. Compute post-match SMD for:
       momentum
       aggression

2. If both absolute SMD <= 0.20:
       keep the pair universe as-is.

3. Otherwise iteratively remove ONE pair at a time.
   At every step evaluate all possible single-pair removals and choose the
   removal that gives the best lexicographic balance improvement:

       lowest max(|momentum SMD|, |aggression SMD|)
       then lowest sum of absolute SMDs
       then remove the pair with larger control distance
       then pair_id as deterministic tie-breaker

4. Stop once both |SMD| <= 0.20, or fewer than 50 pairs remain.

This is a transparent balance-repair/trimming procedure. It does NOT know
whether any candidate later reached 75 or 100 bps.

FROZEN SELECTION RULE
---------------------
For EACH book mechanism separately, after balance repair:

Eligible:
    both post-repair |SMD| <= 0.20
    AND retained pairs >= 50

Among eligible TIGHT / PRIMARY / RELAXED candidates select:

    1. maximum retained pair count
    2. lowest maximum absolute SMD
    3. lowest sum absolute SMD
    4. lowest median pair control distance
    5. caliper name deterministic tie-break

No 75/100 outcomes are loaded before this selection is complete.

PRIMARY SCIENTIFIC QUESTION
---------------------------
After the balanced pair universe is frozen:

    Among comparable causal re-acceleration attempts,
    does the high-liquidity-failure member of a pair reach
    75 bps within 15 minutes more often than the low-failure member?

Secondary:
    100 bps within 15 minutes.

Robustness:
    5 / 10 / 15 / 30 minute outcome horizons
    UP / DOWN diagnostic
    chronological era diagnostic
    candidate-date cluster bootstrap

NO:
* raw depth replay
* raw trade replay
* candidate regeneration
* re-acceleration ontology changes
* composite book-failure score
* ML
* outcome-based match selection
* earlier-than-35 entry search

INPUTS
------
data/derived/perps/research/reacceleration_liquidity_matched_analysis_v1/
    matched_pairs.parquet
    metadata/latest_summary.json

data/derived/perps/research/repeatable_candidate_participant_interaction_v1/
    candidate_decision_outcomes.parquet
    metadata/latest_summary.json

OUTPUTS
-------
data/derived/perps/research/reacceleration_balanced_match_analysis_v1/
    selected_balanced_pairs.parquet       # outcome-blind frozen pair universe
    all_balance_repaired_pairs.parquet    # all candidate repaired universes
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_reacceleration_balanced_match_analysis/
    balance_repair_summary.csv
    balance_repair_trace.csv
    selected_pair_universes.csv
    matched_outcome_contrasts.csv
    horizon_robustness.csv
    direction_robustness.csv
    era_robustness.csv
    day_sign_consistency.csv
    analysis_invariants.csv
    findings.md

INTERPRETATION
--------------
A positive supported effect after the balance gate is stronger evidence that
opposing-liquidity failure carries incremental information in re-accelerations,
rather than residual immediate momentum/aggression explaining the result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "reacceleration_balanced_match_analysis_v1"

FROZEN_PHASE7L4_BUILDER = "1.0.0"
FROZEN_PHASE7L1_BUILDER = "1.0.1"

PRIMARY_OUTCOME_HORIZON_SEC = 900
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
PRIMARY_TARGET_BPS = 75
SECONDARY_TARGET_BPS = 100
PRIMARY_OBSERVATION_HORIZON_SEC = 3

BALANCE_GATE_ABS_SMD = 0.20
MIN_BALANCED_PAIRS = 50

BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260830
MIN_BOOTSTRAP_VALID_REPS = 250

CALIPER_ORDER = ("TIGHT", "PRIMARY", "RELAXED")

CORE_BOOK_FEATURES: tuple[tuple[str, str], ...] = (
    (
        "opposing__durable_unrecovered_loss_share_initial",
        "DURABLE_UNRECOVERED_LOSS",
    ),
    (
        "opposing__peak_initial_queue_loss_share",
        "PEAK_INITIAL_QUEUE_LOSS",
    ),
    (
        "opposing__retained_initial_quote_ratio",
        "LOW_RETAINED_OPPOSING_LIQUIDITY",
    ),
    (
        "support_minus_opposing_retained_ratio",
        "SUPPORT_VS_OPPOSING_ASYMMETRY",
    ),
)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_L4_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "reacceleration_liquidity_matched_analysis_v1"
)
DEFAULT_L1_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "repeatable_candidate_participant_interaction_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_reacceleration_balanced_match_analysis"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bools(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def require(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l41-", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
    )


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l41-", suffix=".csv", dir=str(path.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l41-", suffix=".parquet", dir=str(path.parent)
    )
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_meta(root: Path, builder: str, label: str) -> dict[str, Any]:
    p = root / "metadata" / "latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing {label} metadata: {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if str(m.get("builder_version")) != builder:
        raise RuntimeError(
            f"{label} builder mismatch: expected {builder}, "
            f"found {m.get('builder_version')!r}"
        )
    if str(m.get("status")) != "PASS":
        raise RuntimeError(
            f"{label} must be PASS; found status={m.get('status')!r}"
        )
    return m


def smd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return math.nan
    va = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    den = math.sqrt(max(0.0, 0.5 * (va + vb)))
    if den <= 0:
        return (
            0.0
            if math.isclose(float(np.mean(a)), float(np.mean(b)))
            else math.nan
        )
    return float((np.mean(a) - np.mean(b)) / den)


def balance_metrics(pairs: pd.DataFrame) -> dict[str, float]:
    if pairs.empty:
        return {
            "momentum_smd": math.nan,
            "aggression_smd": math.nan,
            "max_abs_smd": math.nan,
            "sum_abs_smd": math.nan,
        }

    mom = smd(
        pd.to_numeric(pairs["high_mom_z"], errors="coerce").to_numpy(float),
        pd.to_numeric(pairs["low_mom_z"], errors="coerce").to_numpy(float),
    )
    agg = smd(
        pd.to_numeric(pairs["high_agg_z"], errors="coerce").to_numpy(float),
        pd.to_numeric(pairs["low_agg_z"], errors="coerce").to_numpy(float),
    )
    vals = [abs(x) for x in (mom, agg) if math.isfinite(x)]
    return {
        "momentum_smd": mom,
        "aggression_smd": agg,
        "max_abs_smd": max(vals) if vals else math.nan,
        "sum_abs_smd": sum(vals) if vals else math.nan,
    }


def balance_pass(metrics: dict[str, float]) -> bool:
    return bool(
        math.isfinite(metrics["momentum_smd"])
        and math.isfinite(metrics["aggression_smd"])
        and abs(metrics["momentum_smd"]) <= BALANCE_GATE_ABS_SMD
        and abs(metrics["aggression_smd"]) <= BALANCE_GATE_ABS_SMD
    )


def repair_balance(
    pairs: pd.DataFrame,
    *,
    feature: str,
    caliper: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outcome-blind deterministic pair trimming.

    At each iteration test every possible single-pair removal and choose the
    pair whose removal produces the best balance by:

      min max_abs_smd
      min sum_abs_smd
      remove larger original control distance
      pair_id

    Stop when balance passes or dropping another row would move below the
    minimum pair threshold.
    """
    current = pairs.copy().sort_values(
        ["candidate_date", "direction", "pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    trace = []
    step = 0

    while True:
        metrics = balance_metrics(current)
        trace.append({
            "feature": feature,
            "caliper": caliper,
            "step": step,
            "pairs_remaining": int(len(current)),
            **metrics,
            "balance_pass": balance_pass(metrics),
            "action": "STOP_PASS" if balance_pass(metrics) else "EVALUATE_TRIM",
            "removed_pair_id": None,
        })

        if balance_pass(metrics):
            break
        if len(current) <= MIN_BALANCED_PAIRS:
            trace[-1]["action"] = "STOP_MIN_PAIRS"
            break

        choices = []
        for idx, r in current.iterrows():
            candidate = current.drop(index=idx)
            m = balance_metrics(candidate)
            max_abs = (
                safe_float(m["max_abs_smd"])
                if math.isfinite(safe_float(m["max_abs_smd"]))
                else float("inf")
            )
            sum_abs = (
                safe_float(m["sum_abs_smd"])
                if math.isfinite(safe_float(m["sum_abs_smd"]))
                else float("inf")
            )
            distance = safe_float(r.get("control_distance_z"))
            choices.append((
                max_abs,
                sum_abs,
                -distance if math.isfinite(distance) else 0.0,
                str(r["pair_id"]),
                int(idx),
            ))

        choices.sort()
        _, _, _, removed_pair_id, idx = choices[0]
        current = current.drop(index=idx).reset_index(drop=True)

        trace[-1]["action"] = "TRIM"
        trace[-1]["removed_pair_id"] = removed_pair_id
        step += 1

    final = balance_metrics(current)
    current = current.copy()
    current["balance_repair_feature"] = feature
    current["balance_repair_source_caliper"] = caliper
    current["balance_repair_pairs_before"] = int(len(pairs))
    current["balance_repair_pairs_after"] = int(len(current))
    current["balance_repair_trimmed_pairs"] = int(len(pairs) - len(current))
    current["balance_repair_momentum_smd"] = final["momentum_smd"]
    current["balance_repair_aggression_smd"] = final["aggression_smd"]
    current["balance_repair_max_abs_smd"] = final["max_abs_smd"]
    current["balance_repair_pass"] = balance_pass(final)

    return current, pd.DataFrame(trace)


def build_repaired_candidates(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require(
        pairs,
        [
            "pair_id", "feature", "caliper",
            "candidate_date", "direction",
            "high_candidate_id", "low_candidate_id",
            "high_mom_z", "low_mom_z",
            "high_agg_z", "low_agg_z",
            "control_distance_z",
        ],
        "Phase-7L.4 matched_pairs",
    )

    repaired_parts = []
    traces = []
    summaries = []

    for feature, mechanism_label in CORE_BOOK_FEATURES:
        for caliper in CALIPER_ORDER:
            p = pairs[
                (pairs["feature"].astype(str) == feature)
                & (pairs["caliper"].astype(str) == caliper)
            ].copy()

            repaired, trace = repair_balance(
                p,
                feature=feature,
                caliper=caliper,
            )
            metrics = balance_metrics(repaired)
            eligible = bool(
                len(repaired) >= MIN_BALANCED_PAIRS
                and balance_pass(metrics)
            )

            repaired["mechanism_label"] = mechanism_label
            repaired["selection_candidate_eligible"] = eligible
            repaired_parts.append(repaired)
            traces.append(trace)

            summaries.append({
                "feature": feature,
                "mechanism_label": mechanism_label,
                "caliper": caliper,
                "pairs_before_repair": int(len(p)),
                "pairs_after_repair": int(len(repaired)),
                "pairs_trimmed": int(len(p) - len(repaired)),
                **metrics,
                "balance_gate_abs_smd": BALANCE_GATE_ABS_SMD,
                "min_balanced_pairs": MIN_BALANCED_PAIRS,
                "eligible": eligible,
                "median_control_distance_z": (
                    float(
                        pd.to_numeric(
                            repaired["control_distance_z"],
                            errors="coerce",
                        ).median()
                    )
                    if len(repaired) else math.nan
                ),
            })

    all_repaired = pd.concat(repaired_parts, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    summary = pd.DataFrame(summaries)
    return all_repaired, trace, summary


def select_frozen_universes(
    all_repaired: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select pair universe using BALANCE + SAMPLE SIZE only.

    Outcome-free lexicographic rule:
      eligible first;
      maximum retained pairs;
      minimum max abs SMD;
      minimum sum abs SMD;
      minimum median control distance;
      caliper name.
    """
    selected_parts = []
    rows = []

    for feature, mechanism_label in CORE_BOOK_FEATURES:
        x = summary[
            summary["feature"].astype(str) == feature
        ].copy()
        eligible = x[x["eligible"].astype(bool)].copy()

        if eligible.empty:
            rows.append({
                "feature": feature,
                "mechanism_label": mechanism_label,
                "selected": False,
                "selected_caliper": None,
                "pairs": 0,
                "momentum_smd": math.nan,
                "aggression_smd": math.nan,
                "max_abs_smd": math.nan,
                "reason": "NO_ELIGIBLE_BALANCED_UNIVERSE",
            })
            continue

        eligible = eligible.sort_values(
            [
                "pairs_after_repair",
                "max_abs_smd",
                "sum_abs_smd",
                "median_control_distance_z",
                "caliper",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        best = eligible.iloc[0]
        caliper = str(best["caliper"])

        chosen = all_repaired[
            (all_repaired["feature"].astype(str) == feature)
            & (
                all_repaired["balance_repair_source_caliper"].astype(str)
                == caliper
            )
        ].copy()
        chosen["selected_balance_universe"] = True
        chosen["selected_caliper"] = caliper
        selected_parts.append(chosen)

        rows.append({
            "feature": feature,
            "mechanism_label": mechanism_label,
            "selected": True,
            "selected_caliper": caliper,
            "pairs": int(len(chosen)),
            "momentum_smd": safe_float(best["momentum_smd"]),
            "aggression_smd": safe_float(best["aggression_smd"]),
            "max_abs_smd": safe_float(best["max_abs_smd"]),
            "reason": "MAX_PAIRS_THEN_BEST_BALANCE_OUTCOME_BLIND",
        })

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts else
        all_repaired.iloc[0:0].copy()
    )
    selection = pd.DataFrame(rows)
    return selected, selection


def prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    d = outcomes[
        pd.to_numeric(
            outcomes["observation_horizon_sec"], errors="coerce"
        ) == PRIMARY_OBSERVATION_HORIZON_SEC
    ].copy()
    d["candidate_id"] = d["candidate_id"].astype(str)

    if d["candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate primary 3s decision-outcome rows.")

    keep = ["candidate_id", "decision_price_valid"]

    for h in OUTCOME_HORIZONS_SEC:
        cols = [
            f"h{h}_fully_observed",
            f"h{h}_origin_reach_75",
            f"h{h}_origin_reach_100",
        ]
        require(d, cols, "Phase-7L.1 decision outcomes")
        keep.extend(cols)

    x = d[list(dict.fromkeys(keep))].copy()

    for h in OUTCOME_HORIZONS_SEC:
        obs = (
            bools(x["decision_price_valid"])
            & bools(x[f"h{h}_fully_observed"])
        )
        x[f"__obs_{h}"] = obs
        for target in (75, 100):
            x[f"__y_{target}_{h}"] = (
                x[f"h{h}_origin_reach_{target}"]
                .where(obs, pd.NA)
                .astype("boolean")
            )

    return x


def join_outcomes(
    selected_pairs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    hi = outcomes.add_prefix("high__")
    lo = outcomes.add_prefix("low__")

    return (
        selected_pairs
        .merge(
            hi,
            left_on="high_candidate_id",
            right_on="high__candidate_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            lo,
            left_on="low_candidate_id",
            right_on="low__candidate_id",
            how="left",
            validate="many_to_one",
        )
    )


def paired_effect(
    p: pd.DataFrame,
    *,
    target: int,
    horizon: int,
) -> dict[str, Any]:
    hi_obs = bools(p[f"high____obs_{horizon}"])
    lo_obs = bools(p[f"low____obs_{horizon}"])
    x = p[hi_obs & lo_obs].copy()

    hi_y = x[f"high____y_{target}_{horizon}"].astype("boolean")
    lo_y = x[f"low____y_{target}_{horizon}"].astype("boolean")
    valid = hi_y.notna() & lo_y.notna()

    x = x.loc[valid]
    hi_y = hi_y.loc[valid].astype(bool)
    lo_y = lo_y.loc[valid].astype(bool)

    if not len(x):
        return {
            "pairs": 0,
            "high_rate": math.nan,
            "low_rate": math.nan,
            "effect_pp": math.nan,
            "high_wins": 0,
            "low_wins": 0,
            "both_success": 0,
            "both_fail": 0,
        }

    diff = hi_y.astype(int) - lo_y.astype(int)

    return {
        "pairs": int(len(x)),
        "high_rate": float(hi_y.mean()),
        "low_rate": float(lo_y.mean()),
        "effect_pp": 100.0 * float(diff.mean()),
        "high_wins": int((hi_y & ~lo_y).sum()),
        "low_wins": int((~hi_y & lo_y).sum()),
        "both_success": int((hi_y & lo_y).sum()),
        "both_fail": int((~hi_y & ~lo_y).sum()),
    }


def bootstrap_day(
    p: pd.DataFrame,
    *,
    target: int,
    horizon: int,
    reps: int,
    seed: int,
) -> tuple[float, float, int]:
    hi_obs = bools(p[f"high____obs_{horizon}"])
    lo_obs = bools(p[f"low____obs_{horizon}"])
    x = p[hi_obs & lo_obs].copy()

    hi_y = x[f"high____y_{target}_{horizon}"].astype("boolean")
    lo_y = x[f"low____y_{target}_{horizon}"].astype("boolean")
    valid = hi_y.notna() & lo_y.notna()
    x = x.loc[valid].copy()

    if not len(x):
        return math.nan, math.nan, 0

    x["__diff"] = (
        x[f"high____y_{target}_{horizon}"].astype(bool).astype(int)
        - x[f"low____y_{target}_{horizon}"].astype(bool).astype(int)
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
    mult = rng.multinomial(
        n_days,
        np.full(n_days, 1.0 / n_days),
        size=int(reps),
    )
    den = mult @ counts
    vals = np.divide(
        mult @ sums,
        den,
        out=np.full(int(reps), np.nan, dtype=float),
        where=den > 0,
    ) * 100.0
    vals = vals[np.isfinite(vals)]

    if len(vals) < MIN_BOOTSTRAP_VALID_REPS:
        return math.nan, math.nan, int(len(vals))

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def interpretation(effect_pp: float, ci_low: float) -> str:
    if not math.isfinite(effect_pp):
        return "NOT_TESTED"
    if effect_pp <= 0:
        return "NOT_SUPPORTED"
    if math.isfinite(ci_low) and ci_low > 0:
        return "SUPPORTED"
    return "TENTATIVE"


def analyze(
    pair_outcomes: pd.DataFrame,
    *,
    reps: int,
) -> pd.DataFrame:
    rows = []

    for feature, mechanism_label in CORE_BOOK_FEATURES:
        p = pair_outcomes[
            pair_outcomes["feature"].astype(str) == feature
        ].copy()

        for horizon in OUTCOME_HORIZONS_SEC:
            for target in (75, 100):
                r = paired_effect(
                    p,
                    target=target,
                    horizon=horizon,
                )

                if horizon == PRIMARY_OUTCOME_HORIZON_SEC:
                    lo, hi, nboot = bootstrap_day(
                        p,
                        target=target,
                        horizon=horizon,
                        reps=reps,
                        seed=(
                            BOOTSTRAP_SEED
                            + target * 100
                            + sum(ord(c) for c in feature) % 97
                        ),
                    )
                else:
                    lo = hi = math.nan
                    nboot = 0

                rows.append({
                    "feature": feature,
                    "mechanism_label": mechanism_label,
                    "selected_caliper": (
                        str(p["selected_caliper"].iloc[0])
                        if len(p) else None
                    ),
                    "outcome_horizon_sec": horizon,
                    "target_bps": target,
                    **r,
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "bootstrap_valid_reps": nboot,
                    "interpretation": (
                        interpretation(safe_float(r["effect_pp"]), lo)
                        if horizon == PRIMARY_OUTCOME_HORIZON_SEC
                        else "ROBUSTNESS_ONLY"
                    ),
                })

    return pd.DataFrame(rows)


def subgroup_analysis(
    pair_outcomes: pd.DataFrame,
    *,
    kind: str,
    reps: int,
) -> pd.DataFrame:
    rows = []

    if kind == "DIRECTION":
        specs = [
            ("UP", pd.to_numeric(pair_outcomes["direction"], errors="coerce") > 0),
            ("DOWN", pd.to_numeric(pair_outcomes["direction"], errors="coerce") < 0),
        ]
    elif kind == "ERA":
        specs = [
            (
                f"ERA_{era}",
                pd.to_numeric(
                    pair_outcomes["high_era4"], errors="coerce"
                ) == era,
            )
            for era in (1, 2, 3, 4)
        ]
    else:
        raise ValueError(kind)

    for segment, mask in specs:
        d = pair_outcomes.loc[mask].copy()

        for feature, mechanism_label in CORE_BOOK_FEATURES:
            p = d[d["feature"].astype(str) == feature].copy()

            for target in (75, 100):
                r = paired_effect(
                    p,
                    target=target,
                    horizon=PRIMARY_OUTCOME_HORIZON_SEC,
                )

                if kind == "DIRECTION" and len(p):
                    lo, hi, nboot = bootstrap_day(
                        p,
                        target=target,
                        horizon=PRIMARY_OUTCOME_HORIZON_SEC,
                        reps=reps,
                        seed=(
                            BOOTSTRAP_SEED
                            + target * 1000
                            + sum(ord(c) for c in feature + segment) % 193
                        ),
                    )
                else:
                    lo = hi = math.nan
                    nboot = 0

                rows.append({
                    "subgroup_type": kind,
                    "segment": segment,
                    "feature": feature,
                    "mechanism_label": mechanism_label,
                    "target_bps": target,
                    **r,
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "bootstrap_valid_reps": nboot,
                })

    return pd.DataFrame(rows)


def day_sign_consistency(
    pair_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature, mechanism_label in CORE_BOOK_FEATURES:
        p = pair_outcomes[
            pair_outcomes["feature"].astype(str) == feature
        ].copy()

        for target in (75, 100):
            horizon = PRIMARY_OUTCOME_HORIZON_SEC
            x = p[
                bools(p[f"high____obs_{horizon}"])
                & bools(p[f"low____obs_{horizon}"])
            ].copy()

            hi = x[f"high____y_{target}_{horizon}"].astype("boolean")
            lo = x[f"low____y_{target}_{horizon}"].astype("boolean")
            valid = hi.notna() & lo.notna()
            x = x.loc[valid].copy()

            x["__diff"] = (
                x[f"high____y_{target}_{horizon}"].astype(bool).astype(int)
                - x[f"low____y_{target}_{horizon}"].astype(bool).astype(int)
            )

            daily = (
                x.groupby(x["candidate_date"].astype(str))["__diff"].mean()
                if len(x) else
                pd.Series(dtype=float)
            )

            rows.append({
                "feature": feature,
                "mechanism_label": mechanism_label,
                "target_bps": target,
                "comparable_days": int(len(daily)),
                "positive_days": int((daily > 0).sum()),
                "negative_days": int((daily < 0).sum()),
                "zero_days": int((daily == 0).sum()),
                "positive_day_share": (
                    float((daily > 0).mean())
                    if len(daily) else math.nan
                ),
                "median_day_effect_pp": (
                    100.0 * float(daily.median())
                    if len(daily) else math.nan
                ),
            })

    return pd.DataFrame(rows)


def invariants(
    original_pairs: pd.DataFrame,
    all_repaired: pd.DataFrame,
    selected: pd.DataFrame,
    selection: pd.DataFrame,
    pair_outcomes: pd.DataFrame,
    l4_meta: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({
            "invariant": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    outcome_like = [
        c for c in selected.columns
        if (
            "reach_" in c.lower()
            or "outcome" in c.lower()
            or "fully_observed" in c.lower()
            or c.lower().startswith("h300_")
            or c.lower().startswith("h600_")
            or c.lower().startswith("h900_")
            or c.lower().startswith("h1800_")
        )
    ]
    add(
        "selected_pair_manifest_future_clean",
        len(outcome_like) == 0,
        outcome_like,
        [],
    )

    add(
        "outcome_join_preserves_selected_pairs",
        len(pair_outcomes) == len(selected),
        len(pair_outcomes),
        len(selected),
    )

    eligible_selected = selection[
        selection["selected"].astype(bool)
    ]
    bad_balance = int(
        (
            pd.to_numeric(
                eligible_selected["max_abs_smd"], errors="coerce"
            ) > BALANCE_GATE_ABS_SMD + 1e-12
        ).sum()
    )
    add(
        "all_selected_universes_pass_balance_gate",
        bad_balance == 0,
        bad_balance,
        0,
    )

    too_small = int(
        (
            pd.to_numeric(
                eligible_selected["pairs"], errors="coerce"
            ) < MIN_BALANCED_PAIRS
        ).sum()
    )
    add(
        "all_selected_universes_meet_min_pairs",
        too_small == 0,
        too_small,
        0,
    )

    dup_hi = int(
        selected.duplicated(
            ["feature", "high_candidate_id"]
        ).sum()
    ) if len(selected) else 0
    dup_lo = int(
        selected.duplicated(
            ["feature", "low_candidate_id"]
        ).sum()
    ) if len(selected) else 0
    add(
        "selected_without_replacement_high",
        dup_hi == 0,
        dup_hi,
        0,
    )
    add(
        "selected_without_replacement_low",
        dup_lo == 0,
        dup_lo,
        0,
    )

    # Every repaired pair must originate from frozen 7L.4 pair IDs.
    original_ids = set(original_pairs["pair_id"].astype(str))
    repaired_ids = set(all_repaired["pair_id"].astype(str))
    add(
        "all_repaired_pairs_are_subset_of_phase7l4",
        repaired_ids.issubset(original_ids),
        len(repaired_ids - original_ids),
        0,
    )

    add(
        "phase7l4_pairs_were_built_before_outcomes",
        bool(l4_meta.get("pair_manifest_built_before_outcomes", False)),
        l4_meta.get("pair_manifest_built_before_outcomes"),
        True,
    )

    # Feature-level selection should be at most one universe per feature.
    selected_counts = (
        selection[selection["selected"].astype(bool)]
        .groupby("feature").size()
    )
    add(
        "at_most_one_selected_universe_per_feature",
        bool((selected_counts <= 1).all()) if len(selected_counts) else True,
        int((selected_counts > 1).sum()) if len(selected_counts) else 0,
        0,
    )

    return pd.DataFrame(rows)


def findings_text(
    status: str,
    repair_summary: pd.DataFrame,
    selection: pd.DataFrame,
    results: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7L.4.1 — Outcome-blind balance repair + frozen re-acceleration test",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{status}**",
        f"- Balance gate: **|SMD| <= {BALANCE_GATE_ABS_SMD:.2f}** for momentum and aggression",
        f"- Minimum frozen pairs per feature: **{MIN_BALANCED_PAIRS}**",
        "- Match-universe selection used future 75/100 outcomes: **NO**",
        "- Raw depth/trade replay: **NO**",
        "",
        "## Outcome-blind selected pair universes",
        "",
    ]

    for _, r in selection.iterrows():
        if bool(r["selected"]):
            lines.append(
                f"- {r['mechanism_label']}: "
                f"caliper={r['selected_caliper']}, "
                f"pairs={int(r['pairs'])}, "
                f"momentum SMD={safe_float(r['momentum_smd']):+.3f}, "
                f"aggression SMD={safe_float(r['aggression_smd']):+.3f}."
            )
        else:
            lines.append(
                f"- {r['mechanism_label']}: NO balanced >= "
                f"{MIN_BALANCED_PAIRS}-pair universe; not tested for outcomes."
            )

    for target in (75, 100):
        lines += [
            "",
            f"## {target}bp / 15m results on frozen balanced pairs",
            "",
        ]
        x = results[
            (results["target_bps"] == target)
            & (
                results["outcome_horizon_sec"]
                == PRIMARY_OUTCOME_HORIZON_SEC
            )
        ].sort_values("effect_pp", ascending=False)

        for _, r in x.iterrows():
            lines.append(
                f"- {r['mechanism_label']}: "
                f"high={100*safe_float(r['high_rate']):.1f}%, "
                f"low={100*safe_float(r['low_rate']):.1f}%, "
                f"effect={safe_float(r['effect_pp']):+.2f}pp, "
                f"CI=[{safe_float(r['ci_low_pp']):+.2f}, "
                f"{safe_float(r['ci_high_pp']):+.2f}]pp, "
                f"pairs={int(r['pairs'])}, "
                f"**{r['interpretation']}**."
            )

    lines += [
        "",
        "## Discipline",
        "",
        "- This phase fixes the residual-balance weakness identified in 7L.4.",
        "- Pair trimming and feature-level universe selection occurred before outcomes were loaded.",
        "- A feature with no balanced >=50-pair universe is left untested rather than weakening the balance gate.",
        "- This closes the 35bp re-acceleration mechanism branch if the balanced result is interpretable.",
        "- The earlier-than-35bp onset search remains a separate next branch.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> int:
    # Build an imbalanced pair universe that can be repaired outcome-blind.
    rows = []
    for i in range(80):
        # Most pairs nearly balanced.
        hi_m = -0.4 + i * 0.01
        lo_m = hi_m - 0.02
        hi_a = -0.2 + i * 0.005
        lo_a = hi_a - 0.02

        # First 15 pairs create high-side positive imbalance.
        if i < 15:
            hi_m += 1.2
            hi_a += 0.8

        rows.append({
            "pair_id": f"P{i:03d}",
            "feature": "f",
            "caliper": "PRIMARY",
            "candidate_date": f"2026-06-{1 + (i % 10):02d}",
            "direction": 1 if i % 2 == 0 else -1,
            "high_candidate_id": f"H{i}",
            "low_candidate_id": f"L{i}",
            "high_mom_z": hi_m,
            "low_mom_z": lo_m,
            "high_agg_z": hi_a,
            "low_agg_z": lo_a,
            "control_distance_z": abs(hi_m - lo_m) + abs(hi_a - lo_a),
            "high_era4": 1,
            "low_era4": 1,
        })

    p = pd.DataFrame(rows)
    before = balance_metrics(p)
    repaired, trace = repair_balance(
        p,
        feature="f",
        caliper="PRIMARY",
    )
    after = balance_metrics(repaired)

    assert before["max_abs_smd"] > BALANCE_GATE_ABS_SMD
    assert len(repaired) >= MIN_BALANCED_PAIRS
    assert balance_pass(after)
    assert len(repaired) < len(p)
    assert len(trace) >= 2

    # Synthetic selected outcomes.
    po = repaired.copy()
    po["high____obs_900"] = True
    po["low____obs_900"] = True
    po["high____y_75_900"] = [
        i % 4 != 0 for i in range(len(po))
    ]
    po["low____y_75_900"] = [
        i % 4 == 0 for i in range(len(po))
    ]

    eff = paired_effect(po, target=75, horizon=900)
    assert eff["pairs"] == len(po)
    assert eff["effect_pp"] > 0

    lo, hi, nboot = bootstrap_day(
        po,
        target=75,
        horizon=900,
        reps=1_000,
        seed=123,
    )
    assert nboot == 1_000
    assert lo > 0 and hi > 0

    print("SELF-TEST PASS")
    print(
        "Validated outcome-blind iterative balance repair, |SMD| gate, "
        "minimum-pair rule, paired continuation effect, and date-cluster bootstrap."
    )
    return 0


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Phase 7L.4.1 outcome-blind balanced re-acceleration matching"
    )
    p.add_argument(
        "--phase7l4-root",
        type=Path,
        default=DEFAULT_L4_ROOT,
    )
    p.add_argument(
        "--phase7l1-root",
        type=Path,
        default=DEFAULT_L1_ROOT,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    p.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    p.add_argument(
        "--bootstrap-reps",
        type=int,
        default=BOOTSTRAP_REPS,
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    print("Building Phase 7L.4.1 outcome-blind balanced re-acceleration analysis...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Balance gate         : |SMD| <= {BALANCE_GATE_ABS_SMD:.2f}")
    print(f"Minimum pairs        : {MIN_BALANCED_PAIRS}")
    print("Selection rule       : max pairs -> best balance, outcomes hidden")
    print("Raw depth/trade replay: NONE")

    print("[1/7] Loading frozen Phase 7L.4 outcome-blind matched pairs...")
    l4_meta = load_meta(
        args.phase7l4_root,
        FROZEN_PHASE7L4_BUILDER,
        "Phase 7L.4",
    )
    l1_meta = load_meta(
        args.phase7l1_root,
        FROZEN_PHASE7L1_BUILDER,
        "Phase 7L.1",
    )

    pairs_path = args.phase7l4_root / "matched_pairs.parquet"
    outcomes_path = args.phase7l1_root / "candidate_decision_outcomes.parquet"

    if not pairs_path.exists():
        raise RuntimeError(f"Missing Phase 7L.4 matched pairs: {pairs_path}")
    if not outcomes_path.exists():
        raise RuntimeError(f"Missing Phase 7L.1 outcomes: {outcomes_path}")

    original_pairs = pd.read_parquet(pairs_path)
    print(f"      frozen pair rows={len(original_pairs):,}")

    print("[2/7] Repairing balance for every feature x caliper WITHOUT outcomes...")
    all_repaired, repair_trace, repair_summary = build_repaired_candidates(
        original_pairs
    )

    print("[3/7] Freezing best BALANCED pair universe per feature...")
    selected, selection = select_frozen_universes(
        all_repaired,
        repair_summary,
    )

    for _, r in selection.iterrows():
        if bool(r["selected"]):
            print(
                f"      {r['mechanism_label']:34s} "
                f"caliper={str(r['selected_caliper']):8s} "
                f"pairs={int(r['pairs']):3d} "
                f"|SMD|max={safe_float(r['max_abs_smd']):.3f}"
            )
        else:
            print(
                f"      {r['mechanism_label']:34s} "
                "NO ELIGIBLE BALANCED UNIVERSE"
            )

    if selected.empty:
        raise RuntimeError(
            "No feature produced an outcome-blind balanced >=50-pair universe."
        )

    print("[4/7] Loading and joining future outcomes AFTER pair freeze...")
    outcomes = pd.read_parquet(outcomes_path)
    outcome_table = prepare_outcomes(outcomes)
    pair_outcomes = join_outcomes(selected, outcome_table)

    print("[5/7] Computing 75/100 paired effects and robustness...")
    results = analyze(
        pair_outcomes,
        reps=int(args.bootstrap_reps),
    )
    direction = subgroup_analysis(
        pair_outcomes,
        kind="DIRECTION",
        reps=int(args.bootstrap_reps),
    )
    era = subgroup_analysis(
        pair_outcomes,
        kind="ERA",
        reps=int(args.bootstrap_reps),
    )
    day_sign = day_sign_consistency(pair_outcomes)

    print("[6/7] Running invariants...")
    inv = invariants(
        original_pairs,
        all_repaired,
        selected,
        selection,
        pair_outcomes,
        l4_meta,
    )
    failed = inv[inv["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    print("[7/7] Writing atomic outputs...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    atomic_parquet(
        selected,
        args.output_root / "selected_balanced_pairs.parquet",
    )
    atomic_parquet(
        all_repaired,
        args.output_root / "all_balance_repaired_pairs.parquet",
    )
    atomic_csv(
        repair_summary,
        args.report_root / "balance_repair_summary.csv",
    )
    atomic_csv(
        repair_trace,
        args.report_root / "balance_repair_trace.csv",
    )
    atomic_csv(
        selection,
        args.report_root / "selected_pair_universes.csv",
    )
    atomic_csv(
        results,
        args.report_root / "matched_outcome_contrasts.csv",
    )
    atomic_csv(
        results,
        args.report_root / "horizon_robustness.csv",
    )
    atomic_csv(
        direction,
        args.report_root / "direction_robustness.csv",
    )
    atomic_csv(
        era,
        args.report_root / "era_robustness.csv",
    )
    atomic_csv(
        day_sign,
        args.report_root / "day_sign_consistency.csv",
    )
    atomic_csv(
        inv,
        args.report_root / "analysis_invariants.csv",
    )

    primary = results[
        results["outcome_horizon_sec"] == PRIMARY_OUTCOME_HORIZON_SEC
    ].copy()

    supported75 = int(
        (
            (primary["target_bps"] == 75)
            & (primary["interpretation"] == "SUPPORTED")
        ).sum()
    )
    supported100 = int(
        (
            (primary["target_bps"] == 100)
            & (primary["interpretation"] == "SUPPORTED")
        ).sum()
    )

    selected_features = selection[
        selection["selected"].astype(bool)
    ]["feature"].astype(str).tolist()

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "built_at_utc": now_iso(),
        "balance_gate_abs_smd": BALANCE_GATE_ABS_SMD,
        "minimum_balanced_pairs": MIN_BALANCED_PAIRS,
        "candidate_calipers": list(CALIPER_ORDER),
        "selected_features": selected_features,
        "selected_feature_count": int(len(selected_features)),
        "supported_75_features": supported75,
        "supported_100_features": supported100,
        "pair_selection_used_future_outcomes": False,
        "raw_depth_replay": False,
        "raw_trade_replay": False,
        "failed_invariants": int(len(failed)),
        "scientific_state": (
            "REACCELERATION_BALANCED_MATCH_ANALYSIS_COMPLETE_NOT_ALPHA"
        ),
    }

    run_config = {
        **summary,
        "phase7l4_root": str(args.phase7l4_root),
        "phase7l1_root": str(args.phase7l1_root),
        "selection_rule": (
            "eligible if both abs SMD<=0.20 and pairs>=50; "
            "then max pairs, min max abs SMD, min sum abs SMD, "
            "min median distance, caliper name"
        ),
        "repair_rule": (
            "iterative single-pair trimming minimizing max abs SMD, then "
            "sum abs SMD; no future outcomes loaded"
        ),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "phase7l4_pair_manifest_was_outcome_blind": bool(
            l4_meta.get("pair_manifest_built_before_outcomes", False)
        ),
        "phase7l1_builder": l1_meta.get("builder_version"),
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
            status,
            repair_summary,
            selection,
            results,
        ),
    )

    print("")
    print(f"Phase 7L.4.1 status: {status}")
    print(
        f"Balanced features selected: "
        f"{len(selected_features)}/{len(CORE_BOOK_FEATURES)}"
    )
    print(
        f"75bp supported balanced mechanisms: "
        f"{supported75}/{len(selected_features)}"
        if selected_features else
        "75bp supported balanced mechanisms: 0/0"
    )
    print(
        f"100bp supported balanced mechanisms: "
        f"{supported100}/{len(selected_features)}"
        if selected_features else
        "100bp supported balanced mechanisms: 0/0"
    )
    print(f"Reports: {args.report_root}")

    if len(failed):
        print("")
        print(failed.to_string(index=False))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
