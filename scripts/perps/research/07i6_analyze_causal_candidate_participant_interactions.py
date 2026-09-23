#!/usr/bin/env python3
"""
Phase 7I.6 — Causal candidate participant-interaction continuation analysis.

Scientific role
---------------
Cheap retrospective all-market analysis over the frozen Phase-7I.4 substrate.
NO raw depth replay. NO trade replay. NO candidate regeneration.

Primary question
----------------
Among every frozen causal first-35-bp candidate, does the participant interaction
observable three seconds later materially change the probability of continuing to
75 or 100 bps?

Primary predeclared contrast
----------------------------
PRESSURE_PLUS_FAILURE:
    top aggression tercile AND top book-failure tercile
versus
HIGH_PRESSURE_ABSORBED:
    top aggression tercile AND bottom book-failure tercile

This directly asks whether, under similarly high aggressive pressure, a failing
passive book differs from an absorption-like healthy passive book.

No machine learning is trained. Bins/scores are outcome-blind and are computed
separately within each observation horizon. Outcomes are joined only after the
causal score construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

BUILDER_VERSION = "1.0.0"
SCHEMA_NAME = "causal_candidate_participant_interaction_analysis_v1"
SCHEMA_VERSION = 1
FROZEN_PHASE7I4_BUILDER = "1.0.1"
FROZEN_PHASE7I4_SCHEMA = "causal_candidate_participant_interaction_v1"
EXPECTED_CANDIDATES = 2_729
EXPECTED_UP = 1_339
EXPECTED_DOWN = 1_390
EXPECTED_HORIZONS = (1, 3, 5, 10)
PRIMARY_HORIZON_SEC = 3
PRIMARY_OUTCOME_HORIZON_SEC = 1_800
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 20260826
MIN_GROUP_N = 50
MIN_DAY_N = 5

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7I4_SCHEMA
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_causal_candidate_participant_interaction_analysis"

# Positive direction always means "more continuation-favoring" after transformation.
BOOK_FEATURES: tuple[tuple[str, int], ...] = (
    ("opposing__durable_unrecovered_loss_share_initial", +1),
    ("opposing__peak_initial_queue_loss_share", +1),
    ("opposing__retained_initial_quote_ratio", -1),
    ("support_minus_opposing_retained_ratio", +1),
)

AGGRESSION_FEATURES: tuple[tuple[str, int, str], ...] = (
    ("directional_aggressive_quote", +1, "log1p_nonnegative"),
    ("forward_directional_aggressive_imbalance", +1, "identity"),
    ("directional_trade_count_share", +1, "identity"),
    ("directional_net_flow_acceleration_quote_per_sec", +1, "signed_log1p"),
)

CONTEXT_FEATURES: tuple[tuple[str, int], ...] = (
    ("support__retained_initial_quote_ratio", +1),
    ("support__same_price_refill_to_depletion_ratio", +1),
    ("opposing__same_price_refill_to_depletion_ratio", -1),
    ("opposing__durable_same_price_refill_fraction", -1),
    ("largest_trade_share", +1),
    ("trade_size_hhi", +1),
    ("directional_mid_move_bps", +1),
)

FUTURE_PREFIXES = ("h300_", "h600_", "h900_", "h1800_")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return s.astype(str).str.lower().map(mapping).fillna(False).astype(bool)


def sha256_ids(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for x in sorted(str(v) for v in ids):
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def robust_center_scale(values: pd.Series) -> tuple[float, float, str]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(x) == 0:
        return math.nan, math.nan, "NONE"
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return med, 1.4826 * mad, "MAD"
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    if iqr > 1e-12:
        return med, iqr / 1.349, "IQR"
    sd = float(np.std(x, ddof=0))
    if sd > 1e-12:
        return med, sd, "STD"
    return med, 1.0, "CONSTANT"


def transform_feature(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if method == "identity":
        return x
    if method == "log1p_nonnegative":
        return np.log1p(x.clip(lower=0.0))
    if method == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(method)


def percentile_bin(s: pd.Series, n_bins: int) -> pd.Series:
    """Deterministic equal-count bins 1..n_bins; NaN stays nullable Int64."""
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if not mask.any():
        return out
    ranks = x[mask].rank(method="first", pct=True)
    bins = np.ceil(ranks * n_bins).clip(1, n_bins).astype(int)
    out.loc[mask] = bins.astype("Int64")
    return out


def add_outcome_blind_scores(mechanics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = mechanics.copy()
    baseline_rows: list[dict[str, Any]] = []
    df["book_failure_score"] = math.nan
    df["aggression_score"] = math.nan

    for h in EXPECTED_HORIZONS:
        hm = pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == h
        # Book score only on valid exact-book rows.
        book_valid = hm & bool_series(df["window_valid"])
        book_z_cols = []
        for col, sign in BOOK_FEATURES:
            v = pd.to_numeric(df.loc[book_valid, col], errors="coerce") * float(sign)
            center, scale, method = robust_center_scale(v)
            zcol = f"__book_z_{col}"
            df.loc[book_valid, zcol] = ((v - center) / scale).clip(-6, 6) if math.isfinite(scale) and scale > 0 else math.nan
            book_z_cols.append(zcol)
            baseline_rows.append({
                "observation_horizon_sec": h,
                "score_family": "BOOK",
                "feature": col,
                "sign": sign,
                "transform": "identity",
                "center": center,
                "scale": scale,
                "scale_method": method,
                "n": int(v.notna().sum()),
            })
        if book_z_cols:
            df.loc[book_valid, "book_failure_score"] = df.loc[book_valid, book_z_cols].median(axis=1, skipna=True)

        # Trade score is allowed even if the exact book was unavailable.
        aggr_z_cols = []
        for col, sign, transform in AGGRESSION_FEATURES:
            raw = transform_feature(df.loc[hm, col], transform) * float(sign)
            center, scale, method = robust_center_scale(raw)
            zcol = f"__aggr_z_{col}"
            df.loc[hm, zcol] = ((raw - center) / scale).clip(-6, 6) if math.isfinite(scale) and scale > 0 else math.nan
            aggr_z_cols.append(zcol)
            baseline_rows.append({
                "observation_horizon_sec": h,
                "score_family": "AGGRESSION",
                "feature": col,
                "sign": sign,
                "transform": transform,
                "center": center,
                "scale": scale,
                "scale_method": method,
                "n": int(raw.notna().sum()),
            })
        if aggr_z_cols:
            df.loc[hm, "aggression_score"] = df.loc[hm, aggr_z_cols].median(axis=1, skipna=True)

        df.loc[hm, "book_quartile"] = percentile_bin(df.loc[hm, "book_failure_score"], 4).astype(float)
        df.loc[hm, "aggression_quartile"] = percentile_bin(df.loc[hm, "aggression_score"], 4).astype(float)
        df.loc[hm, "book_tercile"] = percentile_bin(df.loc[hm, "book_failure_score"], 3).astype(float)
        df.loc[hm, "aggression_tercile"] = percentile_bin(df.loc[hm, "aggression_score"], 3).astype(float)

    # Drop temporary z columns from the frozen analysis dataset.
    temp = [c for c in df.columns if c.startswith("__book_z_") or c.startswith("__aggr_z_")]
    df = df.drop(columns=temp)
    return df, pd.DataFrame(baseline_rows)


def join_outcomes(scored: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    key = ["candidate_id", "observation_horizon_sec"]
    outcome_cols = [c for c in outcomes.columns if c not in {"builder_version", "schema_version", "day_input_signature"}]
    right = outcomes[outcome_cols].copy()
    # Avoid duplicated causal metadata: keep outcome-only columns plus key.
    duplicate_meta = set(scored.columns) & set(right.columns) - set(key)
    keep = key + [c for c in right.columns if c not in duplicate_meta and c not in key]
    merged = scored.merge(right[keep], on=key, how="left", validate="one_to_one")
    return merged


def first_before(row: pd.Series, fav: int, adv: int, horizon: int = PRIMARY_OUTCOME_HORIZON_SEC) -> Any:
    fh = safe_float(row.get(f"h{horizon}_decision_favorable_{fav}_hit_second"))
    ah = safe_float(row.get(f"h{horizon}_decision_adverse_{adv}_hit_second"))
    fully = bool(row.get(f"h{horizon}_fully_observed", False))
    if math.isfinite(fh) and (not math.isfinite(ah) or fh < ah):
        return True
    if math.isfinite(ah) and (not math.isfinite(fh) or ah < fh):
        return False
    if math.isfinite(fh) and math.isfinite(ah) and fh == ah:
        return None
    return False if fully else None


def prepare_analysis_frame(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    df["candidate_date"] = df["candidate_date"].astype(str)
    for h in EXPECTED_HORIZONS:
        pass
    fully_col = f"h{PRIMARY_OUTCOME_HORIZON_SEC}_fully_observed"
    df["primary_outcome_observed"] = bool_series(df[fully_col]) & bool_series(df["decision_price_valid"])
    for target in (50, 75, 100):
        src = f"h{PRIMARY_OUTCOME_HORIZON_SEC}_origin_reach_{target}"
        df[f"reach_{target}_30m"] = df[src].where(df["primary_outcome_observed"], pd.NA).astype("boolean")
    df["fav20_before_adv10_30m"] = df.apply(lambda r: first_before(r, 20, 10), axis=1).astype("boolean")
    df["fav40_before_adv15_30m"] = df.apply(lambda r: first_before(r, 40, 15), axis=1).astype("boolean")

    # Outcome-blind state definitions.
    df["state"] = "OTHER"
    high_aggr = pd.to_numeric(df["aggression_tercile"], errors="coerce") == 3
    high_fail = pd.to_numeric(df["book_tercile"], errors="coerce") == 3
    low_fail = pd.to_numeric(df["book_tercile"], errors="coerce") == 1
    df.loc[high_aggr & high_fail, "state"] = "PRESSURE_PLUS_FAILURE"
    df.loc[high_aggr & low_fail, "state"] = "HIGH_PRESSURE_ABSORBED"
    df.loc[(~high_aggr) & high_fail, "state"] = "FAILURE_WITHOUT_HIGH_PRESSURE"
    return df


def rate(s: pd.Series) -> float:
    x = s.dropna()
    return float(x.astype(bool).mean()) if len(x) else math.nan


def median_num(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.median()) if len(x) else math.nan


def summarize_group(df: pd.DataFrame, label: str, horizon: int, mask: pd.Series | None = None) -> dict[str, Any]:
    x = df if mask is None else df.loc[mask]
    observed = x[x["primary_outcome_observed"]].copy()
    return {
        "observation_horizon_sec": horizon,
        "group": label,
        "rows": int(len(x)),
        "book_valid_rows": int(bool_series(x["window_valid"]).sum()) if len(x) else 0,
        "outcome_observed_rows": int(len(observed)),
        "p50_30m": rate(observed["reach_50_30m"]) if len(observed) else math.nan,
        "p75_30m": rate(observed["reach_75_30m"]) if len(observed) else math.nan,
        "p100_30m": rate(observed["reach_100_30m"]) if len(observed) else math.nan,
        "median_mfe_30m_bps": median_num(observed[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_mfe_from_decision_bps"]) if len(observed) else math.nan,
        "median_mae_30m_bps": median_num(observed[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_mae_from_decision_bps"]) if len(observed) else math.nan,
        "p_fav20_before_adv10_30m": rate(observed["fav20_before_adv10_30m"]) if len(observed) else math.nan,
        "p_fav40_before_adv15_30m": rate(observed["fav40_before_adv15_30m"]) if len(observed) else math.nan,
        "median_extra_displacement_at_decision_bps": median_num(observed["extra_displacement_since_35bp_candidate_bps"]) if len(observed) else math.nan,
        "median_remaining_to_75_bps": median_num(observed["nominal_remaining_to_75_bps"]) if len(observed) else math.nan,
        "median_remaining_to_100_bps": median_num(observed["nominal_remaining_to_100_bps"]) if len(observed) else math.nan,
    }


def build_base_and_state_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in EXPECTED_HORIZONS:
        d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == h]
        rows.append(summarize_group(d, "ALL_CANDIDATES", h))
        rows.append(summarize_group(d, "BOOK_FAILURE_Q4", h, pd.to_numeric(d["book_quartile"], errors="coerce") == 4))
        rows.append(summarize_group(d, "AGGRESSION_Q4", h, pd.to_numeric(d["aggression_quartile"], errors="coerce") == 4))
        for state in ("PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "FAILURE_WITHOUT_HIGH_PRESSURE"):
            rows.append(summarize_group(d, state, h, d["state"] == state))
    out = pd.DataFrame(rows)
    # Lift relative to the all-candidate base at the same horizon.
    bases = out[out["group"] == "ALL_CANDIDATES"].set_index("observation_horizon_sec")
    for target in (50, 75, 100):
        col = f"p{target}_30m"
        out[f"{col}_abs_lift_vs_base"] = out.apply(
            lambda r: safe_float(r[col]) - safe_float(bases.loc[int(r["observation_horizon_sec"]), col]), axis=1
        )
        out[f"{col}_relative_lift_vs_base"] = out.apply(
            lambda r: safe_float(r[col]) / safe_float(bases.loc[int(r["observation_horizon_sec"]), col])
            if safe_float(bases.loc[int(r["observation_horizon_sec"]), col]) > 0 else math.nan,
            axis=1,
        )
    return out


def build_feature_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features: list[tuple[str, int, str]] = []
    features += [(c, s, "BOOK") for c, s in BOOK_FEATURES]
    features += [(c, s, "AGGRESSION") for c, s, _ in AGGRESSION_FEATURES]
    features += [(c, s, "CONTEXT") for c, s in CONTEXT_FEATURES]
    for h in EXPECTED_HORIZONS:
        d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == h].copy()
        for col, sign, fam in features:
            vals = pd.to_numeric(d[col], errors="coerce") * float(sign)
            bins = percentile_bin(vals, 4)
            for q in (1, 2, 3, 4):
                m = bins == q
                x = d.loc[m & d["primary_outcome_observed"]]
                rows.append({
                    "observation_horizon_sec": h,
                    "family": fam,
                    "feature": col,
                    "continuation_favoring_sign": sign,
                    "quartile": q,
                    "rows": int(m.sum()),
                    "observed_rows": int(len(x)),
                    "median_raw_feature": median_num(d.loc[m, col]),
                    "p75_30m": rate(x["reach_75_30m"]) if len(x) else math.nan,
                    "p100_30m": rate(x["reach_100_30m"]) if len(x) else math.nan,
                    "median_mfe_30m_bps": median_num(x[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_mfe_from_decision_bps"]) if len(x) else math.nan,
                    "median_mae_30m_bps": median_num(x[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_mae_from_decision_bps"]) if len(x) else math.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        piv = out.pivot_table(index=["observation_horizon_sec", "family", "feature"], columns="quartile", values="p75_30m", aggfunc="first")
        diffs = []
        for idx, rr in piv.iterrows():
            q1 = safe_float(rr.get(1))
            q4 = safe_float(rr.get(4))
            diffs.append((*idx, q4 - q1 if math.isfinite(q1) and math.isfinite(q4) else math.nan))
        dd = pd.DataFrame(diffs, columns=["observation_horizon_sec", "family", "feature", "p75_q4_minus_q1"])
        out = out.merge(dd, on=["observation_horizon_sec", "family", "feature"], how="left")
    return out


def build_interaction_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in EXPECTED_HORIZONS:
        d = df[(pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == h) & df["primary_outcome_observed"]].copy()
        for b in (1, 2, 3):
            for a in (1, 2, 3):
                x = d[(pd.to_numeric(d["book_tercile"], errors="coerce") == b) & (pd.to_numeric(d["aggression_tercile"], errors="coerce") == a)]
                rows.append({
                    "observation_horizon_sec": h,
                    "book_failure_tercile": b,
                    "aggression_tercile": a,
                    "rows": int(len(x)),
                    "p75_30m": rate(x["reach_75_30m"]),
                    "p100_30m": rate(x["reach_100_30m"]),
                    "median_mfe_30m_bps": median_num(x[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_mfe_from_decision_bps"]),
                    "median_mae_30m_bps": median_num(x[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_mae_from_decision_bps"]),
                    "p_fav20_before_adv10_30m": rate(x["fav20_before_adv10_30m"]),
                })
    return pd.DataFrame(rows)


def chronology_blocks(dates: pd.Series, n: int = 4) -> dict[str, str]:
    unique = sorted(pd.Series(dates.astype(str).unique()).tolist())
    chunks = np.array_split(np.array(unique, dtype=object), n)
    out: dict[str, str] = {}
    for i, arr in enumerate(chunks, 1):
        for d in arr.tolist():
            out[str(d)] = f"ERA_{i}"
    return out


def paired_state_diff(df: pd.DataFrame, state_a: str, state_b: str, outcome: str) -> tuple[float, int, int]:
    a = df[(df["state"] == state_a) & df["primary_outcome_observed"]]
    b = df[(df["state"] == state_b) & df["primary_outcome_observed"]]
    return rate(a[outcome]) - rate(b[outcome]), int(len(a)), int(len(b))


def build_direction_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]
    for direction_label in ("UP", "DOWN"):
        x = d[d["direction_label"].astype(str) == direction_label]
        diff, na, nb = paired_state_diff(x, "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "reach_75_30m")
        rows.append({
            "subset": direction_label,
            "pressure_failure_n": na,
            "high_pressure_absorbed_n": nb,
            "p75_difference_failure_minus_absorbed": diff,
            "p75_pressure_failure": rate(x[(x["state"] == "PRESSURE_PLUS_FAILURE") & x["primary_outcome_observed"]]["reach_75_30m"]),
            "p75_high_pressure_absorbed": rate(x[(x["state"] == "HIGH_PRESSURE_ABSORBED") & x["primary_outcome_observed"]]["reach_75_30m"]),
        })
    return pd.DataFrame(rows)


def build_chronological_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    cmap = chronology_blocks(d["candidate_date"], 4)
    d["chronology_era"] = d["candidate_date"].astype(str).map(cmap)
    for era, x in d.groupby("chronology_era", sort=True):
        diff, na, nb = paired_state_diff(x, "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "reach_75_30m")
        rows.append({
            "chronology_era": era,
            "start_date": min(x["candidate_date"].astype(str)),
            "end_date": max(x["candidate_date"].astype(str)),
            "candidates": int(x["candidate_id"].nunique()),
            "pressure_failure_n": na,
            "high_pressure_absorbed_n": nb,
            "p75_difference_failure_minus_absorbed": diff,
            "p75_pressure_failure": rate(x[(x["state"] == "PRESSURE_PLUS_FAILURE") & x["primary_outcome_observed"]]["reach_75_30m"]),
            "p75_high_pressure_absorbed": rate(x[(x["state"] == "HIGH_PRESSURE_ABSORBED") & x["primary_outcome_observed"]]["reach_75_30m"]),
        })
    return pd.DataFrame(rows)


def build_day_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df[pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC].copy()
    for day, x in d.groupby("candidate_date", sort=True):
        a = x[(x["state"] == "PRESSURE_PLUS_FAILURE") & x["primary_outcome_observed"]]
        b = x[(x["state"] == "HIGH_PRESSURE_ABSORBED") & x["primary_outcome_observed"]]
        if len(a) < MIN_DAY_N or len(b) < MIN_DAY_N:
            continue
        rows.append({
            "candidate_date": str(day),
            "pressure_failure_n": int(len(a)),
            "high_pressure_absorbed_n": int(len(b)),
            "p75_pressure_failure": rate(a["reach_75_30m"]),
            "p75_high_pressure_absorbed": rate(b["reach_75_30m"]),
            "p75_difference_failure_minus_absorbed": rate(a["reach_75_30m"]) - rate(b["reach_75_30m"]),
        })
    return pd.DataFrame(rows)


def day_cluster_bootstrap_diff(df: pd.DataFrame, state_a: str, state_b: str, outcome: str, reps: int, seed: int) -> dict[str, Any]:
    x = df[(pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC) & df["primary_outcome_observed"]].copy()
    days = sorted(x["candidate_date"].astype(str).unique().tolist())
    observed, na, nb = paired_state_diff(x, state_a, state_b, outcome)
    if not days or not math.isfinite(observed):
        return {"observed_diff": observed, "bootstrap_low": math.nan, "bootstrap_high": math.nan, "reps": 0, "state_a_n": na, "state_b_n": nb}
    grouped = {d: x[x["candidate_date"].astype(str) == d] for d in days}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(reps):
        picked = rng.choice(days, size=len(days), replace=True)
        parts = []
        # Preserve multiplicity of sampled days by concatenating each picked block.
        for d in picked:
            parts.append(grouped[str(d)])
        boot = pd.concat(parts, ignore_index=True)
        diff, _, _ = paired_state_diff(boot, state_a, state_b, outcome)
        if math.isfinite(diff):
            vals.append(diff)
    if not vals:
        lo = hi = math.nan
    else:
        lo, hi = np.percentile(np.array(vals, dtype=float), [2.5, 97.5]).tolist()
    return {
        "observed_diff": observed,
        "bootstrap_low": float(lo),
        "bootstrap_high": float(hi),
        "reps": int(len(vals)),
        "state_a_n": na,
        "state_b_n": nb,
    }


def build_primary_gate(df: pd.DataFrame, direction: pd.DataFrame, chronology: pd.DataFrame, bootstrap: dict[str, Any]) -> pd.DataFrame:
    d = df[(pd.to_numeric(df["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC) & df["primary_outcome_observed"]]
    base = rate(d["reach_75_30m"])
    pf = d[d["state"] == "PRESSURE_PLUS_FAILURE"]
    ab = d[d["state"] == "HIGH_PRESSURE_ABSORBED"]
    p_pf = rate(pf["reach_75_30m"])
    p_ab = rate(ab["reach_75_30m"])
    abs_lift_base = p_pf - base if math.isfinite(p_pf) and math.isfinite(base) else math.nan
    diff_abs = p_pf - p_ab if math.isfinite(p_pf) and math.isfinite(p_ab) else math.nan
    direction_positive = int((pd.to_numeric(direction["p75_difference_failure_minus_absorbed"], errors="coerce") > 0).sum()) if len(direction) else 0
    eras_positive = int((pd.to_numeric(chronology["p75_difference_failure_minus_absorbed"], errors="coerce") > 0).sum()) if len(chronology) else 0

    checks = [
        ("pressure_failure_min_n", len(pf) >= 100, len(pf), ">=100"),
        ("absorbed_min_n", len(ab) >= MIN_GROUP_N, len(ab), f">={MIN_GROUP_N}"),
        ("pressure_failure_abs_lift_vs_base", abs_lift_base >= 0.10 if math.isfinite(abs_lift_base) else False, abs_lift_base, ">=+0.10"),
        ("pressure_failure_vs_absorbed", diff_abs >= 0.10 if math.isfinite(diff_abs) else False, diff_abs, ">=+0.10"),
        ("day_cluster_bootstrap_lower_positive", safe_float(bootstrap.get("bootstrap_low")) > 0, bootstrap.get("bootstrap_low"), ">0"),
        ("up_down_positive", direction_positive == 2, direction_positive, "2/2"),
        ("chronology_positive", eras_positive >= 3, eras_positive, ">=3/4"),
    ]
    return pd.DataFrame([
        {"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected}
        for name, ok, observed, expected in checks
    ])


def build_invariants(mechanics: pd.DataFrame, outcomes: pd.DataFrame, joined: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    key = ["candidate_id", "observation_horizon_sec"]
    add("mechanics_keys_unique", not mechanics.duplicated(key).any(), int(mechanics.duplicated(key).sum()), 0)
    add("outcome_keys_unique", not outcomes.duplicated(key).any(), int(outcomes.duplicated(key).sum()), 0)
    add("joined_keys_unique", not joined.duplicated(key).any(), int(joined.duplicated(key).sum()), 0)
    candidates = mechanics["candidate_id"].astype(str).nunique()
    add("frozen_candidate_count", candidates == EXPECTED_CANDIDATES, candidates, EXPECTED_CANDIDATES)
    primary = mechanics[pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]
    up = int((pd.to_numeric(primary["direction"], errors="coerce") > 0).sum())
    down = int((pd.to_numeric(primary["direction"], errors="coerce") < 0).sum())
    add("frozen_up_down", (up, down) == (EXPECTED_UP, EXPECTED_DOWN), (up, down), (EXPECTED_UP, EXPECTED_DOWN))
    add("four_horizons_present", sorted(pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist()) == list(EXPECTED_HORIZONS), sorted(pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist()), list(EXPECTED_HORIZONS))
    add("four_rows_per_candidate", len(mechanics) == EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS), len(mechanics), EXPECTED_CANDIDATES * len(EXPECTED_HORIZONS))
    coverage = float(bool_series(primary["window_valid"]).mean()) if len(primary) else math.nan
    add("primary_book_coverage", coverage >= 0.95, coverage, ">=0.95")
    source_builders = sorted(mechanics["builder_version"].astype(str).unique().tolist())
    add("frozen_7i4_builder", source_builders == [FROZEN_PHASE7I4_BUILDER], source_builders, [FROZEN_PHASE7I4_BUILDER])
    score_features = [c for c, _ in BOOK_FEATURES] + [c for c, _, _ in AGGRESSION_FEATURES]
    leaking = [c for c in score_features if c.startswith(FUTURE_PREFIXES) or "reach_" in c.lower() or "mfe" in c.lower() or "mae" in c.lower()]
    add("score_features_are_causal_names", len(leaking) == 0, leaking, [])
    scale_ok = bool((pd.to_numeric(baselines["scale"], errors="coerce") > 0).all()) if len(baselines) else False
    add("score_scales_positive", scale_ok, int((pd.to_numeric(baselines["scale"], errors="coerce") <= 0).sum()) if len(baselines) else -1, 0)
    primary_obs = joined[pd.to_numeric(joined["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]
    fully = bool_series(primary_obs[f"h{PRIMARY_OUTCOME_HORIZON_SEC}_fully_observed"]) & bool_series(primary_obs["decision_price_valid"])
    add("primary_outcomes_substantial_coverage", float(fully.mean()) >= 0.95, float(fully.mean()), ">=0.95")
    return pd.DataFrame(rows)


def findings_markdown(summary: dict[str, Any], gate: pd.DataFrame, state_rates: pd.DataFrame, bootstrap: dict[str, Any], direction: pd.DataFrame, chronology: pd.DataFrame) -> str:
    primary = state_rates[state_rates["observation_horizon_sec"] == PRIMARY_HORIZON_SEC].set_index("group")
    def g(group: str, col: str) -> float:
        try:
            return float(primary.loc[group, col])
        except Exception:
            return math.nan
    overall_status = "PASS" if len(gate) and (gate["status"] == "PASS").all() else "FAIL"
    lines = [
        "# Phase 7I.6 — Causal Participant-Interaction Continuation Analysis",
        "",
        "## Status",
        "",
        f"- Builder: `{BUILDER_VERSION}`",
        f"- Engineering status: **{summary['engineering_status']}**",
        f"- Primary scientific interaction gate: **{overall_status}**",
        f"- Primary horizon: **{PRIMARY_HORIZON_SEC}s**",
        f"- Outcome horizon: **{PRIMARY_OUTCOME_HORIZON_SEC // 60} minutes from the actual decision timestamp**",
        f"- Frozen candidates: **{summary['candidate_count']:,}**",
        "",
        "## Primary transparent comparison",
        "",
        "`PRESSURE_PLUS_FAILURE` = top aggression tercile + top book-failure tercile.",
        "",
        "`HIGH_PRESSURE_ABSORBED` = top aggression tercile + bottom book-failure tercile.",
        "",
        f"At 3 seconds, the all-candidate P(75 within 30m) is **{g('ALL_CANDIDATES','p75_30m'):.3f}**.",
        f"The pressure+failure state has P(75) **{g('PRESSURE_PLUS_FAILURE','p75_30m'):.3f}** (N={int(g('PRESSURE_PLUS_FAILURE','outcome_observed_rows')) if math.isfinite(g('PRESSURE_PLUS_FAILURE','outcome_observed_rows')) else 0:,}).",
        f"The high-pressure/absorbed state has P(75) **{g('HIGH_PRESSURE_ABSORBED','p75_30m'):.3f}** (N={int(g('HIGH_PRESSURE_ABSORBED','outcome_observed_rows')) if math.isfinite(g('HIGH_PRESSURE_ABSORBED','outcome_observed_rows')) else 0:,}).",
        f"The pressure+failure minus absorbed P(75) difference is **{safe_float(bootstrap.get('observed_diff')):+.3f}**, with day-cluster bootstrap 95% interval **[{safe_float(bootstrap.get('bootstrap_low')):+.3f}, {safe_float(bootstrap.get('bootstrap_high')):+.3f}]**.",
        "",
        "## Interpretation discipline",
        "",
        "This is an all-market causal retrospective evaluation, not untouched forward validation and not a trading-policy backtest. Scores and bins are outcome-blind, but the May–July 2026 research period has been repeatedly inspected in earlier phases. The correct next step after a robust interaction finding is economic policy analysis and then true forward holdout monitoring—not threshold retuning on this same sample.",
        "",
        "## Gate checks",
        "",
    ]
    for r in gate.to_dict("records"):
        lines.append(f"- {r['check']}: **{r['status']}** — observed `{r['observed']}`, expected `{r['expected']}`")
    lines += ["", "## Direction robustness", ""]
    for r in direction.to_dict("records"):
        lines.append(f"- {r['subset']}: P75 difference failure-minus-absorbed = **{safe_float(r['p75_difference_failure_minus_absorbed']):+.3f}**")
    lines += ["", "## Chronological robustness", ""]
    for r in chronology.to_dict("records"):
        lines.append(f"- {r['chronology_era']} ({r['start_date']} → {r['end_date']}): difference = **{safe_float(r['p75_difference_failure_minus_absorbed']):+.3f}**")
    lines += [
        "",
        "## Files to inspect next",
        "",
        "Start with `participant_interaction_primary_gate.csv`, then `participant_interaction_state_rates.csv`, `participant_interaction_grid.csv`, direction/chronological/day robustness, and finally the economic-outcome columns. Do not select a new threshold merely because one cell is visually largest.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> int:
    # Outcome-blind scoring should rank obviously worse book mechanics higher.
    rows = []
    for i in range(12):
        rows.append({
            "candidate_id": f"c{i}",
            "observation_horizon_sec": 3,
            "window_valid": True,
            "opposing__durable_unrecovered_loss_share_initial": i / 11,
            "opposing__peak_initial_queue_loss_share": i / 11,
            "opposing__retained_initial_quote_ratio": 1 - i / 11,
            "support_minus_opposing_retained_ratio": i / 11,
            "directional_aggressive_quote": float(i + 1) * 1000,
            "forward_directional_aggressive_imbalance": i / 11,
            "directional_trade_count_share": 0.5 + i / 30,
            "directional_net_flow_acceleration_quote_per_sec": float(i - 5),
        })
    # Add other horizons so score function can safely loop.
    base = pd.DataFrame(rows)
    frames = []
    for h in EXPECTED_HORIZONS:
        t = base.copy()
        t["observation_horizon_sec"] = h
        frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    scored, baselines = add_outcome_blind_scores(d)
    p = scored[scored["observation_horizon_sec"] == 3].sort_values("candidate_id")
    assert p["book_failure_score"].iloc[-1] > p["book_failure_score"].iloc[0]
    assert p["aggression_score"].iloc[-1] > p["aggression_score"].iloc[0]
    assert (pd.to_numeric(p["book_tercile"], errors="coerce").dropna().between(1, 3)).all()
    assert (pd.to_numeric(baselines["scale"], errors="coerce") > 0).all()
    # First-passage helper.
    rr = pd.Series({
        "h1800_decision_favorable_20_hit_second": 105,
        "h1800_decision_adverse_10_hit_second": 110,
        "h1800_fully_observed": True,
    })
    assert first_before(rr, 20, 10) is True
    print("SELF-TEST PASS")
    print("Validated outcome-blind robust scoring, deterministic quantile states, and favorable/adverse first-passage ordering.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    mech_path = args.input_root / "candidate_mechanics.parquet"
    out_path = args.input_root / "candidate_decision_outcomes.parquet"
    if not mech_path.exists() or not out_path.exists():
        raise FileNotFoundError(
            "Frozen Phase-7I.4 substrate missing. Expected:\n"
            f"  {mech_path}\n  {out_path}\n"
            "Do not replay raw depth; complete/fix Phase 7I.4 instead."
        )

    print("Building Phase 7I.6 causal participant-interaction continuation analysis...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-7I.4    : {FROZEN_PHASE7I4_BUILDER}")
    print(f"Primary horizon      : {PRIMARY_HORIZON_SEC}s")
    print(f"Outcome horizon      : {PRIMARY_OUTCOME_HORIZON_SEC // 60}m from decision")
    print("Research role        : all-market causal retrospective analysis; no raw replay, no ML")

    print("[1/8] Loading frozen causal mechanics and physically separate outcomes...")
    mechanics = pd.read_parquet(mech_path)
    outcomes = pd.read_parquet(out_path)
    require_columns(mechanics, [
        "candidate_id", "candidate_date", "direction", "direction_label", "observation_horizon_sec",
        "window_valid", "builder_version", "decision_price_valid", "extra_displacement_since_35bp_candidate_bps",
        "nominal_remaining_to_75_bps", "nominal_remaining_to_100_bps",
    ] + [c for c, _ in BOOK_FEATURES] + [c for c, _, _ in AGGRESSION_FEATURES] + [c for c, _ in CONTEXT_FEATURES], "candidate_mechanics")
    require_columns(outcomes, [
        "candidate_id", "observation_horizon_sec", "decision_price_valid",
        "h1800_fully_observed", "h1800_origin_reach_50", "h1800_origin_reach_75", "h1800_origin_reach_100",
        "h1800_mfe_from_decision_bps", "h1800_mae_from_decision_bps",
        "h1800_decision_favorable_20_hit_second", "h1800_decision_adverse_10_hit_second",
        "h1800_decision_favorable_40_hit_second", "h1800_decision_adverse_15_hit_second",
    ], "candidate_decision_outcomes")
    print(f"      mechanics rows={len(mechanics):,} outcomes rows={len(outcomes):,} candidates={mechanics['candidate_id'].nunique():,}")

    print("[2/8] Constructing outcome-blind book-failure and aggressive-pressure scores...")
    scored, score_baselines = add_outcome_blind_scores(mechanics)
    joined = join_outcomes(scored, outcomes)
    analysis = prepare_analysis_frame(joined)

    print("[3/8] Building transparent continuation probabilities and interaction grids...")
    state_rates = build_base_and_state_rates(analysis)
    feature_bins = build_feature_bins(analysis)
    grid = build_interaction_grid(analysis)

    print("[4/8] Auditing direction, chronology, day clustering, and horizon robustness...")
    direction = build_direction_robustness(analysis)
    chronology = build_chronological_robustness(analysis)
    day_robust = build_day_robustness(analysis)
    bootstrap = day_cluster_bootstrap_diff(
        analysis, "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED", "reach_75_30m",
        reps=max(100, int(args.bootstrap_reps)), seed=BOOTSTRAP_SEED,
    )
    gate = build_primary_gate(analysis, direction, chronology, bootstrap)

    print("[5/8] Checking substrate and analysis invariants...")
    invariants = build_invariants(mechanics, outcomes, analysis, score_baselines)
    engineering_pass = bool(len(invariants) and (invariants["status"] == "PASS").all())
    for r in invariants.to_dict("records"):
        print(f"      {r['status']:4s} {r['check']}: observed={r['observed']} expected={r['expected']}")
    if not engineering_pass:
        raise RuntimeError("Phase 7I.6 engineering invariants failed; do not interpret reports.")

    print("[6/8] Writing frozen cheap-analysis dataset and reports...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    # This joined table intentionally contains both causal scores and labels and is analysis-only.
    analysis_out = analysis.copy()
    analysis_out["analysis_builder_version"] = BUILDER_VERSION
    analysis_out["analysis_schema_version"] = SCHEMA_VERSION
    atomic_write_parquet(analysis_out, args.output_root / "participant_interaction_analysis.parquet")
    atomic_write_csv(invariants, args.report_root / "participant_interaction_analysis_invariants.csv")
    atomic_write_csv(score_baselines, args.report_root / "participant_interaction_score_baselines.csv")
    atomic_write_csv(state_rates, args.report_root / "participant_interaction_state_rates.csv")
    atomic_write_csv(feature_bins, args.report_root / "participant_interaction_feature_bins.csv")
    atomic_write_csv(grid, args.report_root / "participant_interaction_grid.csv")
    atomic_write_csv(direction, args.report_root / "participant_interaction_direction_robustness.csv")
    atomic_write_csv(chronology, args.report_root / "participant_interaction_chronological_robustness.csv")
    atomic_write_csv(day_robust, args.report_root / "participant_interaction_day_robustness.csv")
    atomic_write_csv(gate, args.report_root / "participant_interaction_primary_gate.csv")
    atomic_write_csv(pd.DataFrame([bootstrap]), args.report_root / "participant_interaction_day_cluster_bootstrap.csv")

    primary = state_rates[state_rates["observation_horizon_sec"] == PRIMARY_HORIZON_SEC].set_index("group")
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "engineering_status": "PASS",
        "scientific_primary_gate": "PASS" if (gate["status"] == "PASS").all() else "FAIL",
        "frozen_phase7i4_builder": FROZEN_PHASE7I4_BUILDER,
        "candidate_count": int(mechanics["candidate_id"].nunique()),
        "candidate_id_sha256": sha256_ids(mechanics["candidate_id"].astype(str).unique()),
        "primary_horizon_sec": PRIMARY_HORIZON_SEC,
        "outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
        "primary_book_coverage": float(bool_series(mechanics[pd.to_numeric(mechanics["observation_horizon_sec"], errors="coerce") == PRIMARY_HORIZON_SEC]["window_valid"]).mean()),
        "primary_base_p75": safe_float(primary.loc["ALL_CANDIDATES", "p75_30m"]) if "ALL_CANDIDATES" in primary.index else math.nan,
        "primary_pressure_failure_p75": safe_float(primary.loc["PRESSURE_PLUS_FAILURE", "p75_30m"]) if "PRESSURE_PLUS_FAILURE" in primary.index else math.nan,
        "primary_high_pressure_absorbed_p75": safe_float(primary.loc["HIGH_PRESSURE_ABSORBED", "p75_30m"]) if "HIGH_PRESSURE_ABSORBED" in primary.index else math.nan,
        "day_cluster_bootstrap": bootstrap,
        "generated_at": utc_now_iso(),
        "research_role": "all-market causal retrospective participant-interaction analysis; no raw replay; no ML; not untouched forward validation",
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_text(
        args.report_root / "participant_interaction_analysis_findings.md",
        findings_markdown(summary, gate, state_rates, bootstrap, direction, chronology),
    )

    print("[7/8] Primary 3-second interaction summary")
    def show(group: str) -> None:
        if group not in primary.index:
            print(f"      {group:28s} missing")
            return
        r = primary.loc[group]
        print(f"      {group:28s} N={int(r['outcome_observed_rows']):4d} P75={safe_float(r['p75_30m']):.3f} P100={safe_float(r['p100_30m']):.3f} MFE={safe_float(r['median_mfe_30m_bps']):.1f} MAE={safe_float(r['median_mae_30m_bps']):.1f}")
    for group in ("ALL_CANDIDATES", "BOOK_FAILURE_Q4", "AGGRESSION_Q4", "PRESSURE_PLUS_FAILURE", "HIGH_PRESSURE_ABSORBED"):
        show(group)
    print(f"      day-cluster P75 diff failure-absorbed={safe_float(bootstrap.get('observed_diff')):+.3f} 95%=[{safe_float(bootstrap.get('bootstrap_low')):+.3f},{safe_float(bootstrap.get('bootstrap_high')):+.3f}]")

    print("[8/8] Phase 7I.6 summary")
    print(f"      engineering status=PASS")
    print(f"      primary scientific gate={'PASS' if (gate['status'] == 'PASS').all() else 'FAIL'}")
    print(f"      reports={args.report_root}")
    print("      No raw-data replay was performed. Interpret state rates, interaction grid, chronology and direction before any policy optimization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
