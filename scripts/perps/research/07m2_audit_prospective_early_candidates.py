#!/usr/bin/env python3
"""
Phase 7M.2 — Prospective early-candidate audit
=============================================

PLAIN-ENGLISH QUESTION
----------------------
Phase 7M.1 found that acute opposing-side withdrawal around 25 bps was already
informative INSIDE moves that we knew would later reach 35 bps.

That was useful for mechanism timing, but it was not a live early-entry test
because attempts that failed before 35 bps were missing.

Phase 7M.2 fixes that selection problem.

It builds INDEPENDENT causal candidate policies at:

    15 bps  — secondary exploratory threshold
    25 bps  — primary early threshold

using the same frozen state-machine geometry:

    5-bp reset
    10-bp release
    60-second same-direction re-arm

A 15/25-bp candidate is emitted when it actually reaches that threshold live.
It remains in the population regardless of what happens afterward.

Therefore the population includes:

    candidate -> reverses/stalls before 35
    candidate -> reaches 35 but stalls
    candidate -> reaches 50
    candidate -> reaches 75
    candidate -> reaches 100

That is the population required to ask whether 25 bps can become a genuine
earlier examination point.

WHY INDEPENDENT POLICIES?
-------------------------
Do NOT take the old 4,037 35-bp candidates and simply look backward.

The 15-bp and 25-bp policies are regenerated independently from the one-second
price stream. Candidate emission therefore occurs earlier and may change later
re-arm timing/density. This is intentional and is itself audited.

Frozen causal policy per threshold:
    reset        = 5 bps
    release      = 10 bps
    candidate    = 15 OR 25 bps
    re-arm       = 60 seconds

The Phase-7K.1 frozen state machine is reused; only candidate threshold and
60-second re-arm are parameterized.

PRIMARY MECHANISM
-----------------
Phase 7M.1's strongest replicated early signal:

    opposing__peak_1s_withdrawal_frac_3s

For UP candidates:
    opposing side = ASK
For DOWN candidates:
    opposing side = BID

The feature is the largest Phase-6B 1-second opposing withdrawal fraction during
candidate+1 .. candidate+3.

The decision timestamp is candidate+3 seconds.

Diagnostic book proxies are also retained:
    opposing aggregate 5-bp peak loss over 3s
    opposing aggregate 5-bp retained ratio over 3s
    support-minus-opposing retained ratio over 3s

These remain CHEAP Phase-6B aggregate proxies, not exact raw queue replay.

CONTROLS / MATCHING
-------------------
For each threshold separately, BEFORE future outcomes are joined:

    high opposing withdrawal (top tercile)
        matched to
    low opposing withdrawal (bottom tercile)

Exact blocks:
    candidate_date
    direction

Continuous controls:
    3-second post-candidate directional price momentum
    log1p(directional aggressive-quote proxy)

Matching:
    deterministic 1:1 greedy
    without replacement
    per-control z caliper <= 0.75
    Euclidean z distance <= 1.00

Outcome-blind balance repair:
    |momentum SMD|   <= 0.20
    |aggression SMD| <= 0.20

PRE-COMPUTE FEASIBILITY GATE
----------------------------
Before revealing continuation outcomes:

Full prospective population:
    balanced pairs >= 250

Causal re-acceleration diagnostic:
    balanced pairs >= 75

A threshold that cannot satisfy the gate does NOT justify expensive exact-depth
replay.

CAUSAL RE-ACCELERATION CONTEXT
------------------------------
Each early policy receives its own causal re-acceleration label.

At the CURRENT early candidate origin, classify REACCEL_CONTEXT only if:

    origin_kind == REARM_RESET
    AND
    a prior same-direction early candidate had ALREADY confirmed 50 bps
    within the previous 30 minutes
    AND
    no opposite-direction early candidate had confirmed 50 bps after that
    prior same-direction confirmation and before the current origin.

Only historical 50-bp confirmations are allowed.
The current candidate's own future can never classify itself.

OUTCOMES
--------
Future labels are physically separated from causal inputs.

Decision timestamp:
    candidate_second + 3

Primary:
    origin reaches 75 bps within 15 minutes of decision

Secondary:
    origin reaches 35 / 50 / 100 bps within 15 minutes

Robustness:
    75 bps within 5 / 10 / 15 / 30 minutes

Candidate-level base rates are also reported at:
    5 / 10 / 15 / 30 minutes

For the early policies, "failed before 35" is reported transparently as:
    did NOT reach origin 35 bps within the stated horizon.
It is not presented as a permanent physical-leg failure.

DISCOVERY / VALIDATION
----------------------
Retain the existing chronological split:

    DISCOVERY  <= 2026-06-17
    VALIDATION >= 2026-06-18

25 bps is PRIMARY because it replicated most cleanly in 7M.1.
15 bps is SECONDARY / exploratory.

No threshold is selected from outcome significance by this script.

MAPPING TO FROZEN 35-BP UNIVERSE
--------------------------------
For audit only, each frozen 35-bp candidate is greedily mapped one-to-one to the
latest same-direction early candidate whose timestamp lies between that frozen
candidate's release_second and candidate_second.

Report:
    frozen-35 coverage
    early-candidate precision-to-frozen35
    timing lead
    same-origin / same-release agreement

This mapping never creates the early candidate population.

NO:
* raw depth replay
* raw trade replay
* exact queue-level claims
* ML
* composite failure score
* outcome-based threshold tuning

INPUTS
------
scripts/perps/research/07k1_build_causal_repeatable_expansion_impulse_candidate_catalog.py
scripts/perps/research/07m1_screen_early_stage_liquidity_mechanism_onset.py

data/derived/perps/execution_foundation_v1/
data/derived/perps/research/causal_rearm_timing_audit_v1/
data/derived/perps/research/liquidity_event_primitives_v1/

OUTPUTS
-------
data/derived/perps/research/prospective_early_candidate_audit_v1/
    early_candidates.parquet
    early_candidate_inputs.parquet
    matched_pairs.parquet
    early_candidate_outcomes.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_prospective_early_candidate_audit/
    candidate_policy_summary.csv
    candidate_day_profile.csv
    candidate_spacing.csv
    frozen35_mapping_summary.csv
    frozen35_mapping_detail.csv
    precompute_feasibility_gate.csv
    matching_balance.csv
    candidate_base_rates.csv
    withdrawal_matched_outcomes.csv
    discovery_validation.csv
    reacceleration_diagnostic.csv
    analysis_invariants.csv
    findings.md
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import tempfile
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

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
SCHEMA_NAME = "prospective_early_candidate_audit_v1"
DEFAULT_SYMBOL = "BTCUSDT"

FROZEN_PHASE7A_VERSION = "1.0.0"
FROZEN_PHASE7K1_VERSION = "1.0.2"
FROZEN_PHASE7K2_VERSION = "1.0.0"
FROZEN_PHASE7M1_VERSION = "1.0.1"
FROZEN_PHASE6B_VERSION = "1.0.3"

START_DATE = "2026-05-18"
END_DATE = "2026-07-18"
MAX_TRADE_STALE_SEC = 5

RESET_BPS = 5.0
RELEASE_BPS = 10.0
REARM_SEC = 60
EARLY_THRESHOLDS_BPS = (15, 25)
PRIMARY_THRESHOLD_BPS = 25
SECONDARY_THRESHOLD_BPS = 15

POLICY_BY_THRESHOLD = {
    15: "EARLY_15BP_RESET5_RELEASE10_REARM60",
    25: "EARLY_25BP_RESET5_RELEASE10_REARM60",
}

FROZEN_35_POLICY = "TEST_COOLDOWN_60S"
EXPECTED_FROZEN35_CANDIDATES = 4_037

OBSERVATION_HORIZON_SEC = 3
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
TARGETS_BPS = (35, 50, 75, 100)
PRIMARY_TARGET_BPS = 75
PRIMARY_OUTCOME_HORIZON_SEC = 900

PRIMARY_FEATURE = "opposing__peak_1s_withdrawal_frac_3s"
DIAGNOSTIC_FEATURES = (
    "opposing__aggregate5bp_peak_loss_share_3s",
    "opposing__aggregate5bp_retained_ratio_3s",
    "support_minus_opposing__aggregate5bp_retained_ratio_3s",
)

MATCH_Z = 0.75
MATCH_DISTANCE = 1.00
BALANCE_SMD = 0.20
MIN_ALL_BALANCED_PAIRS = 250
MIN_REACCEL_BALANCED_PAIRS = 75
MIN_SPLIT_PAIRS = 100

PRIOR50_LOOKBACK_SEC = 1800

DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"

BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260831
MIN_BOOTSTRAP_VALID_REPS = 250

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_PHASE7A_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07a_build_large_move_event_catalog.py"
)
DEFAULT_PHASE7K1_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07k1_build_causal_repeatable_expansion_impulse_candidate_catalog.py"
)
DEFAULT_PHASE7M1_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07m1_screen_early_stage_liquidity_mechanism_onset.py"
)
DEFAULT_EXECUTION_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
)
DEFAULT_PHASE7K2_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "causal_rearm_timing_audit_v1"
)
DEFAULT_PHASE6B_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "liquidity_event_primitives_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_prospective_early_candidate_audit"
)


def arrow() -> None:
    if pa is None or pq is None:
        raise SystemExit(
            "Phase 7M.2 requires numpy, pandas and pyarrow in the project .venv."
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sf(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bs(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def req(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_meta(root: Path, builder: str, label: str) -> dict[str, Any]:
    p = root / "metadata" / "latest_summary.json"
    if not p.exists():
        raise RuntimeError(f"Missing {label} metadata: {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if str(m.get("builder_version")) != builder:
        raise RuntimeError(
            f"{label} builder expected {builder}; "
            f"found {m.get('builder_version')!r}"
        )
    if str(m.get("status")) != "PASS":
        raise RuntimeError(
            f"{label} metadata status must PASS; found {m.get('status')!r}"
        )
    return m


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m2-", suffix=path.suffix, dir=str(path.parent)
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
        prefix="tmp7m2-", suffix=".csv", dir=str(path.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7m2-", suffix=".parquet", dir=str(path.parent)
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


def load_price(
    p7a,
    execution_root: Path,
    symbol: str,
) -> tuple[np.ndarray, np.ndarray, str, int, int]:
    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    days = []
    d = start - timedelta(days=1)
    while d <= end + timedelta(days=1):
        p = p7a.execution_day_path(execution_root, symbol, d)
        if p.exists():
            days.append(d)
        d += timedelta(days=1)

    if not days:
        raise RuntimeError("No Phase-6A execution partitions found.")

    frame = p7a.load_execution_range(execution_root, symbol, days)
    sec, price, source = p7a.build_catalog_price(
        frame,
        MAX_TRADE_STALE_SEC,
        "last",
    )

    start_epoch = int(
        datetime.combine(
            start, datetime.min.time(), tzinfo=timezone.utc
        ).timestamp()
    )
    end_epoch = int(
        datetime.combine(
            end + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).timestamp()
    )
    del frame
    gc.collect()
    return sec, price, str(source), start_epoch, end_epoch


def generate_early_candidates(
    p7k1,
    sec: np.ndarray,
    price: np.ndarray,
    start_epoch: int,
    end_epoch: int,
) -> pd.DataFrame:
    frames = []

    original_candidate = float(p7k1.CANDIDATE_BPS)
    original_rearm = int(p7k1.MIN_REARM_SEPARATION_SEC)
    original_release = float(p7k1.RELEASE_BPS)

    try:
        p7k1.MIN_REARM_SEPARATION_SEC = int(REARM_SEC)
        p7k1.RELEASE_BPS = float(RELEASE_BPS)

        for threshold in EARLY_THRESHOLDS_BPS:
            p7k1.CANDIDATE_BPS = float(threshold)
            policy = POLICY_BY_THRESHOLD[int(threshold)]

            c = p7k1.build_candidates_for_policy(
                sec,
                price,
                policy,
                RESET_BPS,
                start_epoch,
                end_epoch,
            )
            if len(c):
                c["early_threshold_bps"] = int(threshold)
                c["cooldown_sec"] = int(REARM_SEC)
                c["is_primary_early_threshold"] = (
                    int(threshold) == PRIMARY_THRESHOLD_BPS
                )
                frames.append(c)

            print(
                f"      {threshold:2d}bp policy: "
                f"candidates={len(c):,} "
                f"rearm={(c['origin_kind'].astype(str) == 'REARM_RESET').sum() if len(c) else 0:,}"
            )
    finally:
        p7k1.CANDIDATE_BPS = original_candidate
        p7k1.MIN_REARM_SEPARATION_SEC = original_rearm
        p7k1.RELEASE_BPS = original_release

    if not frames:
        raise RuntimeError("No prospective early candidates generated.")

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["early_threshold_bps", "candidate_second", "direction"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if out["candidate_id"].duplicated().any():
        raise RuntimeError(
            "Prospective early candidate IDs are not globally unique."
        )

    return out


def build_candidate_outcomes(
    p7k1,
    candidates_runtime: pd.DataFrame,
    sec: np.ndarray,
    price: np.ndarray,
) -> pd.DataFrame:
    original_targets = tuple(p7k1.TARGETS_BPS)
    original_max_h = int(p7k1.MAX_OUTCOME_HORIZON_SEC)

    try:
        p7k1.TARGETS_BPS = tuple(float(x) for x in TARGETS_BPS)
        p7k1.MAX_OUTCOME_HORIZON_SEC = int(max(OUTCOME_HORIZONS_SEC))
        out = p7k1.build_candidate_outcomes(
            candidates_runtime,
            sec,
            price,
        )
    finally:
        p7k1.TARGETS_BPS = original_targets
        p7k1.MAX_OUTCOME_HORIZON_SEC = original_max_h

    out = out.merge(
        candidates_runtime[
            ["candidate_id", "early_threshold_bps", "candidate_date"]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    return out


def build_prior50_context(
    candidates: pd.DataFrame,
    candidate_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build causal REACCEL_CONTEXT using only prior 50 confirmations that occurred
    before the CURRENT origin_second.
    """
    c = candidates[
        [
            "candidate_id", "policy_id", "early_threshold_bps",
            "direction", "origin_kind", "origin_second",
            "candidate_second",
        ]
    ].copy()
    o = candidate_outcomes[
        ["candidate_id", "seconds_to_50_from_candidate"]
    ].copy()

    x = c.merge(o, on="candidate_id", how="left", validate="one_to_one")
    x["hit50_second"] = (
        pd.to_numeric(x["candidate_second"], errors="coerce")
        + pd.to_numeric(
            x["seconds_to_50_from_candidate"], errors="coerce"
        )
    )

    context_rows = []

    for threshold in EARLY_THRESHOLDS_BPS:
        policy = POLICY_BY_THRESHOLD[int(threshold)]
        g = x[x["policy_id"].astype(str) == policy].copy()

        indexes: dict[int, dict[str, list[Any]]] = {}
        for direction in (+1, -1):
            h = g[
                (pd.to_numeric(g["direction"], errors="coerce") == direction)
                & g["hit50_second"].notna()
            ].copy()
            h = h.sort_values(
                ["hit50_second", "candidate_id"],
                kind="mergesort",
            )
            indexes[direction] = {
                "seconds": h["hit50_second"].astype(int).tolist(),
                "ids": h["candidate_id"].astype(str).tolist(),
            }

        for r in g.sort_values(
            ["origin_second", "candidate_id"], kind="mergesort"
        ).to_dict("records"):
            direction = int(r["direction"])
            origin = int(r["origin_second"])
            lo = origin - PRIOR50_LOOKBACK_SEC

            same = indexes[direction]["seconds"]
            same_ids = indexes[direction]["ids"]
            j = bisect_right(same, origin) - 1

            latest_same = None
            latest_same_id = None
            if j >= 0 and int(same[j]) >= lo:
                latest_same = int(same[j])
                latest_same_id = str(same_ids[j])

            opposite_after = False
            if latest_same is not None:
                opp = indexes[-direction]["seconds"]
                k = bisect_right(opp, latest_same)
                opposite_after = (
                    k < len(opp) and int(opp[k]) <= origin
                )

            flag = bool(
                str(r["origin_kind"]) == "REARM_RESET"
                and latest_same is not None
                and not opposite_after
            )

            context_rows.append({
                "candidate_id": str(r["candidate_id"]),
                "early_threshold_bps": int(threshold),
                "is_reaccel_context": flag,
                "latest_same_direction_50_hit_30m_second": (
                    latest_same if latest_same is not None else math.nan
                ),
                "latest_same_direction_50_candidate_id": latest_same_id,
                "seconds_since_latest_same_direction_50": (
                    int(origin - latest_same)
                    if latest_same is not None
                    else math.nan
                ),
                "opposite_50_since_latest_same_50": opposite_after,
                "reaccel_context_contract": (
                    "REARM_RESET_AND_PRIOR_SAME50_WITHIN30M_"
                    "NO_OPPOSITE50_BEFORE_CURRENT_ORIGIN"
                ),
            })

    out = pd.DataFrame(context_rows)

    # Current candidate must never be its own prior-50 evidence.
    bad = int(
        (
            out["candidate_id"].astype(str)
            == out["latest_same_direction_50_candidate_id"].astype(str)
        ).sum()
    )
    if bad:
        raise RuntimeError(
            f"Causal reaccel self-leakage detected: {bad} rows."
        )

    return out


def decision_geometry(
    candidates: pd.DataFrame,
    sec: np.ndarray,
    price: np.ndarray,
    price_source: str,
) -> pd.DataFrame:
    rows = []

    for r in candidates.to_dict("records"):
        cs = int(r["candidate_second"])
        ds = cs + OBSERVATION_HORIZON_SEC
        ci = int(np.searchsorted(sec, cs, side="left"))
        di = int(np.searchsorted(sec, ds, side="left"))

        cvalid = (
            ci < len(sec)
            and int(sec[ci]) == cs
            and math.isfinite(float(price[ci]))
            and float(price[ci]) > 0
        )
        dvalid = (
            di < len(sec)
            and int(sec[di]) == ds
            and math.isfinite(float(price[di]))
            and float(price[di]) > 0
        )
        valid = cvalid and dvalid

        cp = float(price[ci]) if cvalid else math.nan
        dp = float(price[di]) if dvalid else math.nan
        direction = int(r["direction"])
        origin = float(r["origin_price"])

        if valid:
            mom = direction * (dp / cp - 1.0) * 10_000.0
            disp = direction * (dp / origin - 1.0) * 10_000.0
        else:
            mom = math.nan
            disp = math.nan

        threshold = int(r["early_threshold_bps"])

        rows.append({
            "candidate_id": str(r["candidate_id"]),
            "policy_id": str(r["policy_id"]),
            "candidate_date": str(r["candidate_date"]),
            "direction": direction,
            "direction_label": str(r["direction_label"]),
            "origin_kind": str(r["origin_kind"]),
            "origin_second": int(r["origin_second"]),
            "origin_price": origin,
            "release_second": int(r["release_second"]),
            "candidate_second": cs,
            "candidate_price": float(r["candidate_price"]),
            "early_threshold_bps": threshold,
            # Phase7M.1 helper consumes stage-shaped rows.
            "stage_bps": threshold,
            "stage_second": cs,
            "stage_date": datetime.fromtimestamp(
                cs, tz=timezone.utc
            ).date().isoformat(),
            "decision_second": ds,
            "proxy_available_after_second": ds + 1,
            "stage_price": cp,
            "decision_price": dp,
            "decision_price_valid": valid,
            "directional_mid_move_bps_3s_proxy": mom,
            "decision_displacement_from_origin_bps": disp,
            "nominal_remaining_stage_to_35_bps": float(35 - threshold),
            "nominal_remaining_stage_to_75_bps": float(75 - threshold),
            "nominal_remaining_stage_to_100_bps": float(100 - threshold),
            "price_source": price_source,
            "future35_conditioned_universe": False,
            "prospective_early_candidate": True,
        })

    return pd.DataFrame(rows)


def build_decision_outcomes(
    candidates: pd.DataFrame,
    sec: np.ndarray,
    price: np.ndarray,
    price_source: str,
) -> pd.DataFrame:
    """
    Physically separate future labels from candidate+3 decision timestamp.
    """
    rows = []

    # Precompute end of contiguous valid segment for each execution index.
    n = len(sec)
    valid_end = np.arange(n, dtype=np.int64)
    last = n - 1
    valid_end[last] = last
    for i in range(n - 2, -1, -1):
        contiguous = (
            int(sec[i + 1]) == int(sec[i]) + 1
            and math.isfinite(float(price[i]))
            and float(price[i]) > 0
            and math.isfinite(float(price[i + 1]))
            and float(price[i + 1]) > 0
        )
        valid_end[i] = valid_end[i + 1] if contiguous else i

    for r in candidates.to_dict("records"):
        cid = str(r["candidate_id"])
        cs = int(r["candidate_second"])
        ds = cs + OBSERVATION_HORIZON_SEC
        di = int(np.searchsorted(sec, ds, side="left"))
        direction = int(r["direction"])
        origin = float(r["origin_price"])

        row: dict[str, Any] = {
            "candidate_id": cid,
            "policy_id": str(r["policy_id"]),
            "early_threshold_bps": int(r["early_threshold_bps"]),
            "candidate_date": str(r["candidate_date"]),
            "direction": direction,
            "direction_label": str(r["direction_label"]),
            "observation_horizon_sec": OBSERVATION_HORIZON_SEC,
            "decision_second": ds,
            "price_source": price_source,
        }

        if (
            di >= len(sec)
            or int(sec[di]) != ds
            or not math.isfinite(float(price[di]))
            or float(price[di]) <= 0
        ):
            row["decision_price_valid"] = False
            row["decision_invalid_reason"] = "DECISION_PRICE_MISSING_OR_STALE"
            rows.append(row)
            continue

        dp = float(price[di])
        disp = direction * (dp / origin - 1.0) * 10_000.0
        row.update({
            "decision_price_valid": True,
            "decision_invalid_reason": None,
            "decision_price": dp,
            "decision_displacement_from_origin_bps": float(disp),
            "nominal_remaining_to_35_bps": float(35.0 - disp),
            "nominal_remaining_to_50_bps": float(50.0 - disp),
            "nominal_remaining_to_75_bps": float(75.0 - disp),
            "nominal_remaining_to_100_bps": float(100.0 - disp),
        })

        seg_end = int(valid_end[di])

        for h in OUTCOME_HORIZONS_SEC:
            target_sec = ds + int(h)
            hi = int(np.searchsorted(sec, target_sec, side="right") - 1)
            fully = (
                hi >= di
                and hi < len(sec)
                and int(sec[hi]) == target_sec
                and seg_end >= hi
            )
            use_hi = (
                min(seg_end, hi)
                if seg_end >= di and hi >= di
                else di
            )

            psec = sec[di: use_hi + 1]
            ppx = price[di: use_hi + 1]
            dr_origin = direction * (ppx / origin - 1.0) * 10_000.0
            dr_decision = direction * (ppx / dp - 1.0) * 10_000.0

            row[f"h{h}_fully_observed"] = bool(fully)
            row[f"h{h}_observed_seconds"] = (
                int(psec[-1] - ds) if len(psec) else 0
            )
            row[f"h{h}_mfe_from_decision_bps"] = (
                float(np.nanmax(dr_decision))
                if len(dr_decision) else math.nan
            )
            row[f"h{h}_mae_from_decision_bps"] = (
                float(max(0.0, -float(np.nanmin(dr_decision))))
                if len(dr_decision) else math.nan
            )

            for target in TARGETS_BPS:
                idx = np.flatnonzero(
                    np.isfinite(dr_origin)
                    & (dr_origin + 1e-12 >= float(target))
                )
                hit = int(psec[int(idx[0])]) if len(idx) else None
                row[f"h{h}_origin_hit_{target}_second"] = (
                    hit if hit is not None else math.nan
                )
                row[f"h{h}_origin_reach_{target}"] = (
                    True
                    if hit is not None
                    else (False if fully else None)
                )
                row[f"h{h}_seconds_to_origin_{target}_from_decision"] = (
                    int(hit - ds) if hit is not None else math.nan
                )

        rows.append(row)

    return pd.DataFrame(rows)


def load_frozen35(k2_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    m = load_meta(
        k2_root,
        FROZEN_PHASE7K2_VERSION,
        "Phase 7K.2",
    )
    p = k2_root / "candidates.parquet"
    if not p.exists():
        raise RuntimeError(f"Missing frozen 35-bp candidates: {p}")
    d = pd.read_parquet(p)
    d = d[d["policy_id"].astype(str) == FROZEN_35_POLICY].copy()
    req(
        d,
        [
            "candidate_id", "direction", "release_second",
            "candidate_second", "origin_second",
        ],
        "Frozen 35-bp candidates",
    )
    if len(d) != EXPECTED_FROZEN35_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_FROZEN35_CANDIDATES:,} frozen 35-bp "
            f"candidates; found {len(d):,}"
        )
    return d.reset_index(drop=True), m


def map_to_frozen35(
    early: pd.DataFrame,
    frozen35: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    summaries = []

    for threshold in EARLY_THRESHOLDS_BPS:
        policy = POLICY_BY_THRESHOLD[int(threshold)]
        e = early[early["policy_id"].astype(str) == policy].copy()

        mapped_early_ids: set[str] = set()
        covered_old = 0

        for direction in (+1, -1):
            eg = e[
                pd.to_numeric(e["direction"], errors="coerce") == direction
            ].sort_values("candidate_second", kind="mergesort")
            old = frozen35[
                pd.to_numeric(
                    frozen35["direction"], errors="coerce"
                ) == direction
            ].sort_values("candidate_second", kind="mergesort")

            esec = eg["candidate_second"].astype(int).to_numpy()
            erecs = eg.to_dict("records")
            used: set[int] = set()

            for r in old.to_dict("records"):
                lo = int(r["release_second"])
                hi = int(r["candidate_second"])

                j = int(np.searchsorted(esec, hi, side="right") - 1)
                chosen = None
                while j >= 0 and int(esec[j]) >= lo:
                    if j not in used:
                        chosen = j
                        break
                    j -= 1

                if chosen is None:
                    detail.append({
                        "early_threshold_bps": int(threshold),
                        "frozen35_candidate_id": str(r["candidate_id"]),
                        "direction": direction,
                        "mapped": False,
                    })
                    continue

                er = erecs[chosen]
                used.add(chosen)
                mapped_early_ids.add(str(er["candidate_id"]))
                covered_old += 1

                detail.append({
                    "early_threshold_bps": int(threshold),
                    "frozen35_candidate_id": str(r["candidate_id"]),
                    "early_candidate_id": str(er["candidate_id"]),
                    "direction": direction,
                    "mapped": True,
                    "early_candidate_second": int(er["candidate_second"]),
                    "frozen35_candidate_second": hi,
                    "lead_seconds": int(hi - int(er["candidate_second"])),
                    "same_origin": bool(
                        int(er["origin_second"]) == int(r["origin_second"])
                    ),
                    "same_release": bool(
                        int(er["release_second"]) == int(r["release_second"])
                    ),
                })

        summaries.append({
            "early_threshold_bps": int(threshold),
            "early_candidates": int(len(e)),
            "frozen35_candidates": int(len(frozen35)),
            "frozen35_covered": int(covered_old),
            "frozen35_coverage_rate": (
                float(covered_old / len(frozen35))
                if len(frozen35) else math.nan
            ),
            "early_candidates_used_in_mapping": int(len(mapped_early_ids)),
            "early_candidate_mapping_share": (
                float(len(mapped_early_ids) / len(e))
                if len(e) else math.nan
            ),
        })

    detail_df = pd.DataFrame(detail)
    summary_df = pd.DataFrame(summaries)

    if len(detail_df):
        for threshold in EARLY_THRESHOLDS_BPS:
            m = detail_df[
                (detail_df["early_threshold_bps"] == threshold)
                & detail_df["mapped"].fillna(False).astype(bool)
            ]
            if len(m):
                idx = summary_df["early_threshold_bps"] == threshold
                summary_df.loc[
                    idx, "median_lead_seconds"
                ] = float(
                    pd.to_numeric(
                        m["lead_seconds"], errors="coerce"
                    ).median()
                )
                summary_df.loc[
                    idx, "same_origin_share"
                ] = float(m["same_origin"].astype(bool).mean())
                summary_df.loc[
                    idx, "same_release_share"
                ] = float(m["same_release"].astype(bool).mean())

    return summary_df, detail_df


def policy_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for threshold in EARLY_THRESHOLDS_BPS:
        g = candidates[
            candidates["early_threshold_bps"].astype(int) == threshold
        ].copy()

        daily = g.groupby(g["candidate_date"].astype(str)).size()

        hour = (
            pd.to_datetime(
                pd.to_numeric(
                    g["candidate_second"], errors="coerce"
                ),
                unit="s",
                utc=True,
            ).dt.floor("h")
        )
        hourly = g.groupby(hour).size()

        rows.append({
            "early_threshold_bps": int(threshold),
            "policy_id": POLICY_BY_THRESHOLD[int(threshold)],
            "candidates": int(len(g)),
            "candidate_days": int(g["candidate_date"].astype(str).nunique()),
            "mean_candidates_per_day": (
                float(daily.mean()) if len(daily) else math.nan
            ),
            "median_candidates_per_day": (
                float(daily.median()) if len(daily) else math.nan
            ),
            "p95_candidates_per_day": (
                float(daily.quantile(0.95)) if len(daily) else math.nan
            ),
            "max_candidates_per_hour": (
                int(hourly.max()) if len(hourly) else 0
            ),
            "up_candidates": int(
                (pd.to_numeric(g["direction"], errors="coerce") > 0).sum()
            ),
            "down_candidates": int(
                (pd.to_numeric(g["direction"], errors="coerce") < 0).sum()
            ),
            "rearm_candidates": int(
                (g["origin_kind"].astype(str) == "REARM_RESET").sum()
            ),
            "rearm_share": float(
                (g["origin_kind"].astype(str) == "REARM_RESET").mean()
            ) if len(g) else math.nan,
        })

    return pd.DataFrame(rows)


def day_profile(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby(
            ["early_threshold_bps", "candidate_date", "direction_label"]
        )
        .size()
        .rename("candidates")
        .reset_index()
        .sort_values(
            ["early_threshold_bps", "candidate_date", "direction_label"],
            kind="mergesort",
        )
    )


def spacing_report(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (threshold, direction), g in candidates.groupby(
        ["early_threshold_bps", "direction"],
        sort=True,
    ):
        sec = np.sort(
            pd.to_numeric(
                g["candidate_second"], errors="coerce"
            ).dropna().astype(np.int64).to_numpy()
        )
        gap = np.diff(sec).astype(float)

        rows.append({
            "early_threshold_bps": int(threshold),
            "direction": int(direction),
            "direction_label": "UP" if int(direction) > 0 else "DOWN",
            "candidates": int(len(sec)),
            "spacing_observations": int(len(gap)),
            "median_spacing_sec": (
                float(np.median(gap)) if len(gap) else math.nan
            ),
            "p10_spacing_sec": (
                float(np.quantile(gap, 0.10)) if len(gap) else math.nan
            ),
            "share_spacing_le_60s": (
                float((gap <= 60).mean()) if len(gap) else math.nan
            ),
            "share_spacing_le_120s": (
                float((gap <= 120).mean()) if len(gap) else math.nan
            ),
            "share_spacing_le_180s": (
                float((gap <= 180).mean()) if len(gap) else math.nan
            ),
        })

    return pd.DataFrame(rows)


def base_rates(
    candidate_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for threshold in EARLY_THRESHOLDS_BPS:
        g = candidate_outcomes[
            candidate_outcomes["early_threshold_bps"].astype(int) == threshold
        ].copy()

        for h_min in (5, 10, 15, 30):
            row = {
                "early_threshold_bps": int(threshold),
                "horizon_min": int(h_min),
                "candidates": int(len(g)),
            }
            for target in TARGETS_BPS:
                col = f"reach_{target}_within_{h_min}m"
                if col not in g.columns:
                    # Phase7K1 names integer targets.
                    col = f"reach_{int(target)}_within_{h_min}m"
                row[f"p_reach_{target}"] = (
                    float(bs(g[col]).mean()) if len(g) else math.nan
                )
            row["p_fail_to_reach_35"] = (
                1.0 - row["p_reach_35"]
                if math.isfinite(sf(row["p_reach_35"]))
                else math.nan
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_match_universes(
    inputs: pd.DataFrame,
    p7m1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_parts = []
    balance_rows = []
    counter = 0

    populations = {
        "ALL": pd.Series(True, index=inputs.index),
        "REACCEL_CONTEXT": bs(inputs["is_reaccel_context"]),
    }

    for threshold in EARLY_THRESHOLDS_BPS:
        tmask = inputs["early_threshold_bps"].astype(int) == threshold

        for population, pmask in populations.items():
            min_pairs = (
                MIN_ALL_BALANCED_PAIRS
                if population == "ALL"
                else MIN_REACCEL_BALANCED_PAIRS
            )

            base = inputs[
                tmask
                & pmask
                & bs(inputs["proxy_valid"])
                & inputs["__momentum_z"].notna()
                & inputs["__aggression_z"].notna()
            ].copy()

            tercile = f"__tercile__{PRIMARY_FEATURE}"
            x = base[
                pd.to_numeric(
                    base[tercile], errors="coerce"
                ).isin([1, 3])
            ].copy()

            high = x[
                pd.to_numeric(
                    x[tercile], errors="coerce"
                ) == 3
            ]
            low = x[
                pd.to_numeric(
                    x[tercile], errors="coerce"
                ) == 1
            ]

            rows = []
            for (_, _), block in x.groupby(
                ["candidate_date", "direction"],
                sort=True,
            ):
                bh = block[
                    pd.to_numeric(
                        block[tercile], errors="coerce"
                    ) == 3
                ]
                bl = block[
                    pd.to_numeric(
                        block[tercile], errors="coerce"
                    ) == 1
                ]
                rows.extend(
                    p7m1.greedy(
                        bh,
                        bl,
                        PRIMARY_FEATURE,
                        int(threshold),
                        population,
                    )
                )

            raw = pd.DataFrame(rows)
            if len(raw):
                raw["pair_id"] = [
                    f"7M2-{counter + i + 1:08d}"
                    for i in range(len(raw))
                ]
                counter += len(raw)

            pre = p7m1.balance(raw)
            repaired = p7m1.repair(raw, min_pairs)
            post = p7m1.balance(repaired)
            feasible = bool(
                len(repaired) >= min_pairs
                and p7m1.bal_ok(post)
            )

            if len(repaired):
                repaired["early_threshold_bps"] = int(threshold)
                repaired["mechanism_label"] = (
                    "OPPOSING_PEAK_1S_WITHDRAWAL_3S"
                )
                repaired["pairs_before_repair"] = int(len(raw))
                repaired["pairs_after_repair"] = int(len(repaired))
                repaired["final_momentum_smd"] = post["momentum_smd"]
                repaired["final_aggression_smd"] = post["aggression_smd"]
                repaired["final_balance_pass"] = p7m1.bal_ok(post)
                pair_parts.append(repaired)

            balance_rows.append({
                "early_threshold_bps": int(threshold),
                "population": population,
                "mechanism": "OPPOSING_PEAK_1S_WITHDRAWAL_3S",
                "population_candidates": int(len(base)),
                "high_tercile_candidates": int(len(high)),
                "low_tercile_candidates": int(len(low)),
                "matched_pairs_before_repair": int(len(raw)),
                "matched_pairs_after_repair": int(len(repaired)),
                "pairs_trimmed": int(len(raw) - len(repaired)),
                "pre_repair_momentum_smd": pre["momentum_smd"],
                "pre_repair_aggression_smd": pre["aggression_smd"],
                "post_repair_momentum_smd": post["momentum_smd"],
                "post_repair_aggression_smd": post["aggression_smd"],
                "post_repair_max_abs_smd": post["max_abs_smd"],
                "balance_gate_abs_smd": BALANCE_SMD,
                "min_pairs_gate": int(min_pairs),
                "feasible": feasible,
            })

    return (
        pd.concat(pair_parts, ignore_index=True)
        if pair_parts else pd.DataFrame(),
        pd.DataFrame(balance_rows),
    )


def join_pair_outcomes(
    pairs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    hi = outcomes.add_prefix("high__")
    lo = outcomes.add_prefix("low__")
    return (
        pairs
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
    if p.empty:
        return {
            "pairs": 0,
            "high_rate": math.nan,
            "low_rate": math.nan,
            "effect_pp": math.nan,
            "high_wins": 0,
            "low_wins": 0,
        }

    obs = (
        bs(p[f"high__h{horizon}_fully_observed"])
        & bs(p[f"low__h{horizon}_fully_observed"])
        & bs(p["high__decision_price_valid"])
        & bs(p["low__decision_price_valid"])
    )
    x = p.loc[obs].copy()

    hcol = f"high__h{horizon}_origin_reach_{target}"
    lcol = f"low__h{horizon}_origin_reach_{target}"

    hy = x[hcol].astype("boolean")
    ly = x[lcol].astype("boolean")
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

    return {
        "pairs": int(len(x)),
        "high_rate": float(hy.mean()),
        "low_rate": float(ly.mean()),
        "effect_pp": 100.0 * float(
            (hy.astype(int) - ly.astype(int)).mean()
        ),
        "high_wins": int((hy & ~ly).sum()),
        "low_wins": int((~hy & ly).sum()),
    }


def bootstrap_day(
    p: pd.DataFrame,
    *,
    target: int,
    horizon: int,
    reps: int,
    seed: int,
) -> tuple[float, float, int]:
    obs = (
        bs(p[f"high__h{horizon}_fully_observed"])
        & bs(p[f"low__h{horizon}_fully_observed"])
        & bs(p["high__decision_price_valid"])
        & bs(p["low__decision_price_valid"])
    )
    x = p.loc[obs].copy()
    if not len(x):
        return math.nan, math.nan, 0

    hcol = f"high__h{horizon}_origin_reach_{target}"
    lcol = f"low__h{horizon}_origin_reach_{target}"

    hy = x[hcol].astype("boolean")
    ly = x[lcol].astype("boolean")
    valid = hy.notna() & ly.notna()
    x = x.loc[valid].copy()

    if not len(x):
        return math.nan, math.nan, 0

    x["__diff"] = (
        x[hcol].astype(bool).astype(int)
        - x[lcol].astype(bool).astype(int)
    )
    g = (
        x.groupby(x["candidate_date"].astype(str))["__diff"]
        .agg(["sum", "count"])
    )
    if len(g) < 2:
        return math.nan, math.nan, 0

    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    n = len(g)

    rng = np.random.default_rng(seed)
    mult = rng.multinomial(
        n,
        np.full(n, 1.0 / n),
        size=int(reps),
    )
    den = mult @ counts
    vals = np.divide(
        mult @ sums,
        den,
        out=np.full(int(reps), np.nan),
        where=den > 0,
    ) * 100.0
    vals = vals[np.isfinite(vals)]

    if len(vals) < MIN_BOOTSTRAP_VALID_REPS:
        return math.nan, math.nan, int(len(vals))

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def analyze_matched(
    pair_outcomes: pd.DataFrame,
    balance: pd.DataFrame,
    *,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    split_rows = []
    reacc_rows = []

    feasible = {
        (int(r.early_threshold_bps), str(r.population)): bool(r.feasible)
        for r in balance.itertuples(index=False)
    }

    for threshold in EARLY_THRESHOLDS_BPS:
        for population in ("ALL", "REACCEL_CONTEXT"):
            p = pair_outcomes[
                (pair_outcomes["early_threshold_bps"].astype(int) == threshold)
                & (pair_outcomes["population"].astype(str) == population)
            ].copy()

            is_feasible = feasible.get((threshold, population), False)

            # Primary/secondary targets at 15m.
            for target in TARGETS_BPS:
                r = paired_effect(
                    p,
                    target=target,
                    horizon=PRIMARY_OUTCOME_HORIZON_SEC,
                )
                if is_feasible:
                    lo, hi, nboot = bootstrap_day(
                        p,
                        target=target,
                        horizon=PRIMARY_OUTCOME_HORIZON_SEC,
                        reps=reps,
                        seed=(
                            BOOTSTRAP_SEED
                            + threshold * 1000
                            + target * 10
                            + (50_000 if population == "REACCEL_CONTEXT" else 0)
                        ),
                    )
                else:
                    lo = hi = math.nan
                    nboot = 0

                row = {
                    "early_threshold_bps": int(threshold),
                    "population": population,
                    "target_bps": int(target),
                    "outcome_horizon_sec": PRIMARY_OUTCOME_HORIZON_SEC,
                    "feasible_before_outcomes": is_feasible,
                    **r,
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "bootstrap_valid_reps": nboot,
                }
                rows.append(row)
                if population == "REACCEL_CONTEXT":
                    reacc_rows.append(row)

            # 75-bp timing robustness.
            for h in OUTCOME_HORIZONS_SEC:
                if h == PRIMARY_OUTCOME_HORIZON_SEC:
                    continue
                r = paired_effect(
                    p,
                    target=PRIMARY_TARGET_BPS,
                    horizon=h,
                )
                rows.append({
                    "early_threshold_bps": int(threshold),
                    "population": population,
                    "target_bps": PRIMARY_TARGET_BPS,
                    "outcome_horizon_sec": int(h),
                    "feasible_before_outcomes": is_feasible,
                    **r,
                    "ci_low_pp": math.nan,
                    "ci_high_pp": math.nan,
                    "bootstrap_valid_reps": 0,
                })

            # Discovery/validation only for full population / P75 15m.
            if population == "ALL":
                for split, mask in (
                    (
                        "DISCOVERY",
                        p["candidate_date"].astype(str)
                        <= DISCOVERY_END_DATE,
                    ),
                    (
                        "VALIDATION",
                        p["candidate_date"].astype(str)
                        >= VALIDATION_START_DATE,
                    ),
                ):
                    sp = p.loc[mask].copy()
                    r = paired_effect(
                        sp,
                        target=PRIMARY_TARGET_BPS,
                        horizon=PRIMARY_OUTCOME_HORIZON_SEC,
                    )
                    split_ok = (
                        is_feasible and len(sp) >= MIN_SPLIT_PAIRS
                    )
                    if split_ok:
                        lo, hi, nboot = bootstrap_day(
                            sp,
                            target=PRIMARY_TARGET_BPS,
                            horizon=PRIMARY_OUTCOME_HORIZON_SEC,
                            reps=reps,
                            seed=(
                                BOOTSTRAP_SEED
                                + threshold * 100
                                + sum(ord(c) for c in split)
                            ),
                        )
                    else:
                        lo = hi = math.nan
                        nboot = 0

                    split_rows.append({
                        "early_threshold_bps": int(threshold),
                        "split": split,
                        "pairs_available": int(len(sp)),
                        "min_split_pairs": MIN_SPLIT_PAIRS,
                        "split_feasible": split_ok,
                        **r,
                        "ci_low_pp": lo,
                        "ci_high_pp": hi,
                        "bootstrap_valid_reps": nboot,
                    })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(reacc_rows),
    )


def checks(
    candidates_public: pd.DataFrame,
    candidate_inputs: pd.DataFrame,
    candidate_outcomes: pd.DataFrame,
    frozen35: pd.DataFrame,
    pairs: pd.DataFrame,
    balance: pd.DataFrame,
    pair_outcomes: pd.DataFrame,
    price_source: str,
    metas: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    def add(name: str, ok: bool, obs: Any, exp: Any) -> None:
        rows.append({
            "invariant": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "observed": obs,
            "expected": exp,
        })

    add(
        "price_source",
        price_source == f"LAST_TRADE_FFILL_MAX_{MAX_TRADE_STALE_SEC}S",
        price_source,
        f"LAST_TRADE_FFILL_MAX_{MAX_TRADE_STALE_SEC}S",
    )

    observed_thresholds = sorted(
        pd.to_numeric(
            candidates_public["early_threshold_bps"], errors="coerce"
        ).dropna().astype(int).unique().tolist()
    )
    add(
        "early_thresholds_exact",
        observed_thresholds == list(EARLY_THRESHOLDS_BPS),
        observed_thresholds,
        list(EARLY_THRESHOLDS_BPS),
    )

    add(
        "reset5",
        bool(
            (
                pd.to_numeric(
                    candidates_public["reset_threshold_bps"],
                    errors="coerce",
                ) == RESET_BPS
            ).all()
        ),
        sorted(
            pd.to_numeric(
                candidates_public["reset_threshold_bps"],
                errors="coerce",
            ).dropna().unique().tolist()
        ),
        [RESET_BPS],
    )
    add(
        "release10",
        bool(
            (
                pd.to_numeric(
                    candidates_public["release_threshold_bps"],
                    errors="coerce",
                ) == RELEASE_BPS
            ).all()
        ),
        sorted(
            pd.to_numeric(
                candidates_public["release_threshold_bps"],
                errors="coerce",
            ).dropna().unique().tolist()
        ),
        [RELEASE_BPS],
    )
    add(
        "rearm60",
        bool(
            (
                pd.to_numeric(
                    candidates_public["cooldown_sec"],
                    errors="coerce",
                ) == REARM_SEC
            ).all()
        ),
        sorted(
            pd.to_numeric(
                candidates_public["cooldown_sec"],
                errors="coerce",
            ).dropna().unique().tolist()
        ),
        [REARM_SEC],
    )

    # Causal input manifest must not contain current candidate future targets.
    future_tokens = (
        "reach_35",
        "reach_50",
        "reach_75",
        "reach_100",
        "seconds_to_35",
        "seconds_to_50",
        "seconds_to_75",
        "seconds_to_100",
        "outcome",
    )
    leaks = [
        c for c in candidate_inputs.columns
        if any(t in c.lower() for t in future_tokens)
    ]
    add(
        "candidate_inputs_future_clean",
        len(leaks) == 0,
        leaks,
        [],
    )

    pair_leaks = [
        c for c in pairs.columns
        if (
            "reach_" in c.lower()
            or "outcome" in c.lower()
            or "fully_observed" in c.lower()
        )
    ]
    add(
        "matched_pairs_future_clean",
        len(pair_leaks) == 0,
        pair_leaks,
        [],
    )

    bad_balance = int(
        (
            balance["feasible"].astype(bool)
            & (
                pd.to_numeric(
                    balance["post_repair_max_abs_smd"],
                    errors="coerce",
                ) > BALANCE_SMD + 1e-12
            )
        ).sum()
    )
    add(
        "all_feasible_pairs_balanced",
        bad_balance == 0,
        bad_balance,
        0,
    )

    add(
        "outcome_join_preserves_pairs",
        len(pair_outcomes) == len(pairs),
        len(pair_outcomes),
        len(pairs),
    )
    add(
        "frozen35_count",
        len(frozen35) == EXPECTED_FROZEN35_CANDIDATES,
        len(frozen35),
        EXPECTED_FROZEN35_CANDIDATES,
    )

    add(
        "candidate_outcomes_one_to_one",
        len(candidate_outcomes) == len(candidates_public),
        len(candidate_outcomes),
        len(candidates_public),
    )

    for label, expected in (
        ("phase7k2", FROZEN_PHASE7K2_VERSION),
        ("phase6b", FROZEN_PHASE6B_VERSION),
    ):
        add(
            f"{label}_builder",
            str(metas[label].get("builder_version")) == expected,
            metas[label].get("builder_version"),
            expected,
        )

    return pd.DataFrame(rows)


def findings(
    status: str,
    summary: pd.DataFrame,
    mapping: pd.DataFrame,
    feasibility: pd.DataFrame,
    base: pd.DataFrame,
    matched: pd.DataFrame,
    split: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7M.2 — Prospective early-candidate audit",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{status}**",
        "- Primary threshold: **25bp**",
        "- Secondary threshold: **15bp**",
        "- Candidate policies are independent causal state machines.",
        "- Failed-before-35 attempts are included.",
        "- Raw depth/trade replay: **NO**",
        "",
        "## Candidate populations",
        "",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"- {int(r['early_threshold_bps'])}bp: "
            f"{int(r['candidates']):,} candidates; "
            f"median/day={sf(r['median_candidates_per_day']):.1f}; "
            f"peak/hour={int(r['max_candidates_per_hour'])}; "
            f"rearm share={100*sf(r['rearm_share']):.1f}%."
        )

    lines += ["", "## Mapping back to frozen 35bp universe", ""]
    for _, r in mapping.iterrows():
        lines.append(
            f"- {int(r['early_threshold_bps'])}bp: "
            f"covers {int(r['frozen35_covered']):,}/"
            f"{int(r['frozen35_candidates']):,} frozen35 "
            f"({100*sf(r['frozen35_coverage_rate']):.1f}%); "
            f"only {100*sf(r['early_candidate_mapping_share']):.1f}% of "
            f"early candidates are consumed by that mapping."
        )

    lines += ["", "## Pre-compute withdrawal matching feasibility", ""]
    for _, r in feasibility.iterrows():
        lines.append(
            f"- {int(r['early_threshold_bps'])}bp / {r['population']}: "
            f"balanced pairs={int(r['matched_pairs_after_repair'])}; "
            f"momentum SMD={sf(r['post_repair_momentum_smd']):+.3f}; "
            f"aggression SMD={sf(r['post_repair_aggression_smd']):+.3f}; "
            f"feasible={bool(r['feasible'])}."
        )

    lines += ["", "## Prospective base rates — 15-minute horizon", ""]
    b15 = base[base["horizon_min"] == 15]
    for _, r in b15.iterrows():
        lines.append(
            f"- {int(r['early_threshold_bps'])}bp: "
            f"P35={100*sf(r['p_reach_35']):.1f}%, "
            f"P50={100*sf(r['p_reach_50']):.1f}%, "
            f"P75={100*sf(r['p_reach_75']):.1f}%, "
            f"P100={100*sf(r['p_reach_100']):.1f}%; "
            f"fail-to-35={100*sf(r['p_fail_to_reach_35']):.1f}%."
        )

    lines += [
        "",
        "## Opposing-withdrawal matched 75bp / 15m result",
        "",
    ]
    x = matched[
        (matched["population"] == "ALL")
        & (matched["target_bps"] == 75)
        & (
            matched["outcome_horizon_sec"]
            == PRIMARY_OUTCOME_HORIZON_SEC
        )
    ]
    for _, r in x.iterrows():
        lines.append(
            f"- {int(r['early_threshold_bps'])}bp: "
            f"high withdrawal={100*sf(r['high_rate']):.1f}%, "
            f"low withdrawal={100*sf(r['low_rate']):.1f}%, "
            f"effect={sf(r['effect_pp']):+.2f}pp, "
            f"CI=[{sf(r['ci_low_pp']):+.2f}, {sf(r['ci_high_pp']):+.2f}]pp, "
            f"pairs={int(r['pairs'])}, "
            f"feasible-before-outcomes={bool(r['feasible_before_outcomes'])}."
        )

    lines += [
        "",
        "## Interpretation discipline",
        "",
        "- This phase is the first prospective 15/25bp candidate test.",
        "- The candidate population was created before and independently of 35/50/75/100 outcomes.",
        "- A positive 25bp withdrawal effect now includes failed-before-35 attempts.",
        "- Exact raw-depth replay is justified only if the 25bp prospective population passes the pre-compute gate and the effect reproduces in validation.",
        "- 15bp remains exploratory even if its in-sample effect is attractive.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> int:
    # Causal prior-50 classification: current candidate cannot classify itself.
    candidates = pd.DataFrame([
        {
            "candidate_id": "a",
            "policy_id": POLICY_BY_THRESHOLD[25],
            "early_threshold_bps": 25,
            "direction": 1,
            "origin_kind": "INITIAL_EXTREME",
            "origin_second": 100,
            "candidate_second": 120,
        },
        {
            "candidate_id": "b",
            "policy_id": POLICY_BY_THRESHOLD[25],
            "early_threshold_bps": 25,
            "direction": 1,
            "origin_kind": "REARM_RESET",
            "origin_second": 240,
            "candidate_second": 260,
        },
        {
            "candidate_id": "c",
            "policy_id": POLICY_BY_THRESHOLD[25],
            "early_threshold_bps": 25,
            "direction": -1,
            "origin_kind": "INITIAL_EXTREME",
            "origin_second": 300,
            "candidate_second": 320,
        },
        {
            "candidate_id": "d",
            "policy_id": POLICY_BY_THRESHOLD[25],
            "early_threshold_bps": 25,
            "direction": 1,
            "origin_kind": "REARM_RESET",
            "origin_second": 400,
            "candidate_second": 420,
        },
    ])
    outcomes = pd.DataFrame([
        {"candidate_id": "a", "seconds_to_50_from_candidate": 40},
        {"candidate_id": "b", "seconds_to_50_from_candidate": math.nan},
        {"candidate_id": "c", "seconds_to_50_from_candidate": 20},
        {"candidate_id": "d", "seconds_to_50_from_candidate": math.nan},
    ])

    ctx = build_prior50_context(candidates, outcomes).set_index("candidate_id")
    assert bool(ctx.loc["b", "is_reaccel_context"])
    assert not bool(ctx.loc["d", "is_reaccel_context"])
    assert not bool(ctx.loc["a", "is_reaccel_context"])

    # One-to-one early->frozen35 mapping.
    early = pd.DataFrame([
        {
            "candidate_id": "e1",
            "policy_id": POLICY_BY_THRESHOLD[25],
            "early_threshold_bps": 25,
            "direction": 1,
            "candidate_second": 120,
            "origin_second": 100,
            "release_second": 110,
        },
        {
            "candidate_id": "e2",
            "policy_id": POLICY_BY_THRESHOLD[25],
            "early_threshold_bps": 25,
            "direction": 1,
            "candidate_second": 220,
            "origin_second": 200,
            "release_second": 210,
        },
        {
            "candidate_id": "e3",
            "policy_id": POLICY_BY_THRESHOLD[15],
            "early_threshold_bps": 15,
            "direction": 1,
            "candidate_second": 115,
            "origin_second": 100,
            "release_second": 110,
        },
    ])
    frozen = pd.DataFrame([
        {
            "candidate_id": "f1",
            "direction": 1,
            "release_second": 110,
            "candidate_second": 130,
            "origin_second": 100,
        },
        {
            "candidate_id": "f2",
            "direction": 1,
            "release_second": 210,
            "candidate_second": 240,
            "origin_second": 200,
        },
    ])
    sm, det = map_to_frozen35(early, frozen)
    s25 = sm[sm["early_threshold_bps"] == 25].iloc[0]
    assert int(s25["frozen35_covered"]) == 2

    # Balance/matching primitives.
    p = pd.DataFrame({
        "high_momentum_z": np.linspace(-0.2, 0.2, 300),
        "low_momentum_z": np.linspace(-0.21, 0.19, 300),
        "high_aggression_z": np.linspace(-0.1, 0.1, 300),
        "low_aggression_z": np.linspace(-0.11, 0.09, 300),
    })
    m = {
        "momentum_smd": smd_for_test(
            p["high_momentum_z"].to_numpy(),
            p["low_momentum_z"].to_numpy(),
        ),
        "aggression_smd": smd_for_test(
            p["high_aggression_z"].to_numpy(),
            p["low_aggression_z"].to_numpy(),
        ),
    }
    assert abs(m["momentum_smd"]) <= BALANCE_SMD
    assert abs(m["aggression_smd"]) <= BALANCE_SMD

    print("SELF-TEST PASS")
    print(
        "Validated causal prior-50 reacceleration context without self-leakage, "
        "independent early-to-frozen35 mapping, and prospective balance thresholds."
    )
    return 0


def smd_for_test(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    den = math.sqrt(0.5 * (va + vb))
    return float((a.mean() - b.mean()) / den)


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Phase 7M.2 prospective early-candidate audit"
    )
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument(
        "--phase7a-script",
        type=Path,
        default=DEFAULT_PHASE7A_SCRIPT,
    )
    p.add_argument(
        "--phase7k1-script",
        type=Path,
        default=DEFAULT_PHASE7K1_SCRIPT,
    )
    p.add_argument(
        "--phase7m1-script",
        type=Path,
        default=DEFAULT_PHASE7M1_SCRIPT,
    )
    p.add_argument(
        "--execution-root",
        type=Path,
        default=DEFAULT_EXECUTION_ROOT,
    )
    p.add_argument(
        "--phase7k2-root",
        type=Path,
        default=DEFAULT_PHASE7K2_ROOT,
    )
    p.add_argument(
        "--phase6b-root",
        type=Path,
        default=DEFAULT_PHASE6B_ROOT,
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
    a = parse_args(argv)

    if a.self_test:
        return self_test()

    arrow()

    if a.symbol != DEFAULT_SYMBOL:
        raise SystemExit(
            f"Phase 7M.2 is frozen to {DEFAULT_SYMBOL}; received {a.symbol}"
        )

    print("Building Phase 7M.2 prospective early-candidate audit...")
    print(f"Builder version         : {BUILDER_VERSION}")
    print("Primary / secondary     : 25bp / 15bp")
    print("Reset / release / rearm : 5bp / 10bp / 60s")
    print("Primary mechanism       : opposing peak 1s withdrawal over candidate+3s")
    print("Failed-before-35 attempts: INCLUDED")
    print("Raw depth/trade replay  : NONE")

    print("[1/10] Loading frozen scripts/contracts...")
    p7a = load_module(a.phase7a_script, "phase7a_for_7m2")
    p7k1 = load_module(a.phase7k1_script, "phase7k1_for_7m2")
    p7m1 = load_module(a.phase7m1_script, "phase7m1_for_7m2")

    if str(getattr(p7a, "BUILDER_VERSION", "")) != FROZEN_PHASE7A_VERSION:
        raise RuntimeError(
            f"Requires Phase7A {FROZEN_PHASE7A_VERSION}; "
            f"found {getattr(p7a, 'BUILDER_VERSION', None)!r}"
        )
    if str(getattr(p7k1, "BUILDER_VERSION", "")) != FROZEN_PHASE7K1_VERSION:
        raise RuntimeError(
            f"Requires Phase7K1 {FROZEN_PHASE7K1_VERSION}; "
            f"found {getattr(p7k1, 'BUILDER_VERSION', None)!r}"
        )
    if str(getattr(p7m1, "BUILDER_VERSION", "")) != FROZEN_PHASE7M1_VERSION:
        raise RuntimeError(
            f"Requires Phase7M1 {FROZEN_PHASE7M1_VERSION}; "
            f"found {getattr(p7m1, 'BUILDER_VERSION', None)!r}"
        )

    phase6b_meta = load_meta(
        a.phase6b_root,
        FROZEN_PHASE6B_VERSION,
        "Phase 6B",
    )
    frozen35, k2_meta = load_frozen35(a.phase7k2_root)

    print("[2/10] Loading frozen one-second execution price...")
    sec, price, price_source, start_epoch, end_epoch = load_price(
        p7a,
        a.execution_root,
        a.symbol,
    )
    print(
        f"      rows={len(sec):,}; source={price_source}; "
        f"window={START_DATE}..{END_DATE}"
    )

    print("[3/10] Generating INDEPENDENT prospective 15/25bp policies...")
    early_runtime = generate_early_candidates(
        p7k1,
        sec,
        price,
        start_epoch,
        end_epoch,
    )

    print("[4/10] Building prior-50 history ledger/context labels...")
    # Candidate-timestamp outcome table is temporarily built here solely so a
    # prior candidate's 50 confirmation can be recognized when it is already
    # historical at a later candidate's origin. Current-candidate outcome data
    # are not joined into causal inputs.
    candidate_outcomes = build_candidate_outcomes(
        p7k1,
        early_runtime,
        sec,
        price,
    )
    context = build_prior50_context(
        early_runtime,
        candidate_outcomes,
    )

    early_runtime = early_runtime.merge(
        context,
        on=["candidate_id", "early_threshold_bps"],
        how="left",
        validate="one_to_one",
    )
    early_runtime["is_reaccel_context"] = bs(
        early_runtime["is_reaccel_context"]
    )

    for threshold in EARLY_THRESHOLDS_BPS:
        g = early_runtime[
            early_runtime["early_threshold_bps"].astype(int) == threshold
        ]
        print(
            f"      {threshold:2d}bp causal reaccel context: "
            f"{int(g['is_reaccel_context'].sum()):,}/{len(g):,}"
        )

    print("[5/10] Building candidate+3s CHEAP Phase6B inputs...")
    geometry = decision_geometry(
        early_runtime,
        sec,
        price,
        price_source,
    )
    inputs = p7m1.stage_proxies(
        geometry,
        a.phase6b_root,
        a.symbol,
    )
    inputs = inputs.merge(
        early_runtime[
            ["candidate_id", "is_reaccel_context"]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    inputs["is_reaccel_context"] = bs(inputs["is_reaccel_context"])

    valid = int(bs(inputs["proxy_valid"]).sum())
    print(
        f"      valid proxy rows={valid:,}/{len(inputs):,} "
        f"({100*valid/max(1,len(inputs)):.2f}%)"
    )

    print("[6/10] PRE-COMPUTE matching/feasibility BEFORE outcomes...")
    pairs, balance = build_match_universes(inputs, p7m1)

    for _, r in balance.iterrows():
        print(
            f"      {int(r['early_threshold_bps']):2d}bp "
            f"{str(r['population']):17s}: "
            f"pairs={int(r['matched_pairs_after_repair']):4d} "
            f"SMD=({sf(r['post_repair_momentum_smd']):+.3f},"
            f"{sf(r['post_repair_aggression_smd']):+.3f}) "
            f"FEASIBLE={bool(r['feasible'])}"
        )

    # Freeze all causal artifacts before future decision outcomes are created.
    candidates_public = early_runtime.drop(
        columns=["_price_index"],
        errors="ignore",
    ).copy()

    causal_drop = [
        c for c in inputs.columns
        if (
            c.startswith("latest_same_direction_50")
            or c.startswith("seconds_since_latest_same_direction_50")
        )
    ]
    # Historical context metadata is causal, but keep only the final flag in the
    # live input manifest to avoid confusion with outcome-style timestamps.
    inputs_public = inputs.drop(
        columns=causal_drop,
        errors="ignore",
    ).copy()

    a.output_root.mkdir(parents=True, exist_ok=True)
    a.report_root.mkdir(parents=True, exist_ok=True)

    atomic_parquet(
        candidates_public,
        a.output_root / "early_candidates.parquet",
    )
    atomic_parquet(
        inputs_public,
        a.output_root / "early_candidate_inputs.parquet",
    )
    atomic_parquet(
        pairs,
        a.output_root / "matched_pairs.parquet",
    )
    atomic_csv(
        balance,
        a.report_root / "precompute_feasibility_gate.csv",
    )
    atomic_csv(
        balance,
        a.report_root / "matching_balance.csv",
    )

    print("[7/10] Auditing density/redundancy and frozen35 mapping...")
    pol_summary = policy_summary(candidates_public)
    days = day_profile(candidates_public)
    spacing = spacing_report(candidates_public)
    map_summary, map_detail = map_to_frozen35(
        candidates_public,
        frozen35,
    )

    atomic_csv(
        pol_summary,
        a.report_root / "candidate_policy_summary.csv",
    )
    atomic_csv(
        days,
        a.report_root / "candidate_day_profile.csv",
    )
    atomic_csv(
        spacing,
        a.report_root / "candidate_spacing.csv",
    )
    atomic_csv(
        map_summary,
        a.report_root / "frozen35_mapping_summary.csv",
    )
    atomic_csv(
        map_detail,
        a.report_root / "frozen35_mapping_detail.csv",
    )

    print("[8/10] Creating physically separate future outcomes AFTER gate freeze...")
    decision_outcomes = build_decision_outcomes(
        early_runtime,
        sec,
        price,
        price_source,
    )
    atomic_parquet(
        decision_outcomes,
        a.output_root / "early_candidate_outcomes.parquet",
    )

    base = base_rates(candidate_outcomes)
    atomic_csv(
        base,
        a.report_root / "candidate_base_rates.csv",
    )

    print("[9/10] Testing prospective opposing-withdrawal continuation...")
    pair_outcomes = join_pair_outcomes(
        pairs,
        decision_outcomes,
    )
    matched, split, reacc = analyze_matched(
        pair_outcomes,
        balance,
        reps=int(a.bootstrap_reps),
    )

    atomic_csv(
        matched,
        a.report_root / "withdrawal_matched_outcomes.csv",
    )
    atomic_csv(
        split,
        a.report_root / "discovery_validation.csv",
    )
    atomic_csv(
        reacc,
        a.report_root / "reacceleration_diagnostic.csv",
    )

    print("[10/10] Running invariants + writing metadata...")
    inv = checks(
        candidates_public,
        inputs_public,
        candidate_outcomes,
        frozen35,
        pairs,
        balance,
        pair_outcomes,
        price_source,
        {
            "phase7k2": k2_meta,
            "phase6b": phase6b_meta,
        },
    )
    failed = inv[inv["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    atomic_csv(
        inv,
        a.report_root / "analysis_invariants.csv",
    )
    atomic_text(
        a.report_root / "findings.md",
        findings(
            status,
            pol_summary,
            map_summary,
            balance,
            base,
            matched,
            split,
        ),
    )

    primary_gate = balance[
        (balance["early_threshold_bps"].astype(int) == PRIMARY_THRESHOLD_BPS)
        & (balance["population"] == "ALL")
    ]
    primary_feasible = bool(
        len(primary_gate)
        and primary_gate.iloc[0]["feasible"]
    )

    primary_validation = split[
        (split["early_threshold_bps"].astype(int) == PRIMARY_THRESHOLD_BPS)
        & (split["split"] == "VALIDATION")
    ]
    validation_effect = (
        sf(primary_validation.iloc[0]["effect_pp"])
        if len(primary_validation) else math.nan
    )
    validation_ci_low = (
        sf(primary_validation.iloc[0]["ci_low_pp"])
        if len(primary_validation) else math.nan
    )

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "built_at_utc": now_iso(),
        "symbol": a.symbol,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "price_source": price_source,
        "candidate_thresholds_bps": list(EARLY_THRESHOLDS_BPS),
        "primary_threshold_bps": PRIMARY_THRESHOLD_BPS,
        "secondary_threshold_bps": SECONDARY_THRESHOLD_BPS,
        "reset_bps": RESET_BPS,
        "release_bps": RELEASE_BPS,
        "rearm_sec": REARM_SEC,
        "candidate_count_by_threshold": {
            str(t): int(
                (
                    candidates_public["early_threshold_bps"].astype(int)
                    == t
                ).sum()
            )
            for t in EARLY_THRESHOLDS_BPS
        },
        "reaccel_context_count_by_threshold": {
            str(t): int(
                candidates_public.loc[
                    candidates_public["early_threshold_bps"].astype(int) == t,
                    "is_reaccel_context",
                ].astype(bool).sum()
            )
            for t in EARLY_THRESHOLDS_BPS
        },
        "primary_25bp_precompute_feasible": primary_feasible,
        "primary_25bp_validation_effect_pp": validation_effect,
        "primary_25bp_validation_ci_low_pp": validation_ci_low,
        "prospective_population_includes_failed_before35": True,
        "pair_manifests_frozen_before_decision_outcomes": True,
        "raw_depth_replay": False,
        "raw_trade_replay": False,
        "failed_invariants": int(len(failed)),
        "scientific_state": (
            "PROSPECTIVE_EARLY_CANDIDATE_AUDIT_COMPLETE_NOT_ALPHA"
        ),
    }

    run_config = {
        **summary,
        "phase7a_script": str(a.phase7a_script),
        "phase7k1_script": str(a.phase7k1_script),
        "phase7m1_script": str(a.phase7m1_script),
        "execution_root": str(a.execution_root),
        "phase7k2_root": str(a.phase7k2_root),
        "phase6b_root": str(a.phase6b_root),
        "primary_feature": PRIMARY_FEATURE,
        "matching": {
            "exact_blocks": ["candidate_date", "direction"],
            "controls": [
                "directional_mid_move_bps_3s_proxy",
                "log1p(directional_aggressive_quote_5s_proxy)",
            ],
            "per_control_z_caliper": MATCH_Z,
            "euclidean_z_caliper": MATCH_DISTANCE,
            "balance_smd_gate": BALANCE_SMD,
            "min_all_pairs": MIN_ALL_BALANCED_PAIRS,
            "min_reaccel_pairs": MIN_REACCEL_BALANCED_PAIRS,
        },
        "reaccel_context_contract": (
            "REARM_RESET + prior same-direction confirmed50 within30m + "
            "no opposite confirmed50 before current origin"
        ),
        "discovery_end_date": DISCOVERY_END_DATE,
        "validation_start_date": VALIDATION_START_DATE,
        "bootstrap_reps": int(a.bootstrap_reps),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    atomic_json(
        a.output_root / "metadata" / "latest_summary.json",
        summary,
    )
    atomic_json(
        a.output_root / "metadata" / "run_config.json",
        run_config,
    )

    print("")
    print(f"Phase 7M.2 status: {status}")
    print(f"25bp pre-compute feasible: {primary_feasible}")
    print(
        "25bp validation P75 withdrawal effect: "
        f"{validation_effect:+.2f}pp "
        f"(CI low {validation_ci_low:+.2f}pp)"
        if math.isfinite(validation_effect)
        else "25bp validation P75 withdrawal effect: NOT TESTED"
    )
    print(f"Reports: {a.report_root}")

    if len(failed):
        print("")
        print(failed.to_string(index=False))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
