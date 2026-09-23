#!/usr/bin/env python3
"""
Phase 7L.3 — Causal re-acceleration context + feasibility audit
================================================================

PLAIN-ENGLISH QUESTION
----------------------
Can we label the frozen 60-second causal candidates as occurring in a
re-acceleration context using ONLY information already known before the current
candidate, and will that label leave enough observations to run a proper
liquidity-mechanism analysis?

This phase exists because the earlier Phase-7K.4 FIRST/REACCEL association is a
hindsight onset audit. It is useful for structural validation, but it is not a
complete live-context label: candidates that looked like re-acceleration
attempts but later failed to become a common-50 impulse can remain UNASSOCIATED.

CRITICAL EXISTING CAUSAL STATE
------------------------------
The frozen Phase-7K.2 / Phase-7K.1 candidate generator already stores:

    origin_kind == INITIAL_EXTREME
    origin_kind == REARM_RESET

REARM_RESET is causal. It means that after a previous same-direction candidate,
the state machine observed the frozen cooldown/reset/release geometry and formed
a fresh origin before the current 35-bp candidate.

Therefore this phase does NOT invent re-acceleration from future outcomes.
Instead it audits several increasingly strict causal context definitions.

DEFINITIONS AUDITED
-------------------
A. REARM_STATE
   Current candidate has origin_kind == REARM_RESET.

B. REARM_IMMEDIATE_PRIOR50
   REARM_STATE, AND the immediately preceding same-direction candidate had
   already reached 50 bp from its own origin BEFORE the current reset origin.

C. REARM_ANY_PRIOR50_30M
   REARM_STATE, AND at least one earlier same-direction candidate had already
   reached 50 bp within the previous 30 minutes before the current reset origin.

D. REARM_ANY_PRIOR50_30M_NO_OPPOSITE50
   Definition C, AND no opposite-direction candidate had reached 50 bp between
   that prior same-direction 50-bp confirmation and the current reset origin.

Why 50 bp?
----------
The frozen ontology now treats 50 bp as "an expansion impulse exists", with
75 bp as substantial continuation and 100 bp as major continuation. Requiring a
PRIOR 50-bp confirmation therefore asks whether a real earlier expansion had
already happened before the current causal reset/release/candidate.

CAUSAL USE OF THE OUTCOME TABLE
-------------------------------
Phase-7K.2 candidate_outcomes.parquet stores each old candidate's first-passage
timing. We use ONLY a prior candidate's 50-bp hit when that hit timestamp is
<= the CURRENT candidate's origin timestamp.

That event is historical information by the current decision time.

The CURRENT candidate's own 50/75/100 outcome is NEVER read to classify itself.
No 75/100 outcome is used anywhere in definition selection or feasibility.

STRUCTURAL AUDIT
----------------
Frozen Phase-7K.4 candidate_onset_associations.parquet provides confident
hindsight FIRST/REACCEL labels for a subset.

For that subset only, report:
* reacceleration recall
* FIRST false-positive rate
* specificity
* precision among structurally-labelled candidates
* balanced accuracy
* confusion counts

UNASSOCIATED candidates are NOT treated as structural negatives.

PRE-COMPUTE FEASIBILITY GATE
----------------------------
Before any downstream re-acceleration mechanism analysis, use the ALREADY-BUILT
Phase-7L.1 3-second mechanics to simulate the exact planned stratification:

    momentum tercile
      x directional-aggressive-notional tercile
      x high/low individual book-mechanism tercile

For each re-acceleration definition and each of the four core book-failure
features, count common support using the same rule as Phase 7L.2:

    a cell is usable only if:
        high-book N >= 10
        low-book  N >= 10

    common_support_N = 2 * sum(min(N_high, N_low))

Feasibility threshold for the future PRIMARY all-reacceleration analysis:
    common_support_N >= 250
for at least 3 of the 4 core book mechanisms.

Direction-specific support is reported separately with a diagnostic threshold
of 125, but does not automatically block the pooled re-acceleration analysis.

IMPORTANT:
This phase does not run the 75/100 mechanism comparison itself. It only answers:
    "Can we define the subgroup causally?"
    "Is the subgroup large enough for the planned controlled test?"

NO:
* raw depth replay
* raw trade replay
* candidate regeneration
* ML
* composite book-failure score
* profitability analysis
* current-candidate continuation-rate optimization

INPUTS
------
data/derived/perps/research/causal_rearm_timing_audit_v1/
    candidates.parquet
    candidate_outcomes.parquet
    metadata/latest_summary.json

data/derived/perps/research/candidate_redundancy_earliest_availability_audit_v1/
    candidate_onset_associations.parquet
    metadata/latest_summary.json

data/derived/perps/research/repeatable_candidate_participant_interaction_v1/
    candidate_mechanics.parquet
    metadata/latest_summary.json

OUTPUTS
-------
data/derived/perps/research/causal_reacceleration_context_audit_v1/
    candidate_context_labels.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_causal_reacceleration_context_audit/
    context_definition_counts.csv
    structural_agreement.csv
    feasibility_summary.csv
    feasibility_cell_detail.csv
    context_timing_profile.csv
    analysis_invariants.csv
    findings.md

INTERPRETATION DISCIPLINE
-------------------------
No definition is automatically frozen. The report ranks structurally-audited,
feasible definitions for review, without consulting 75/100 continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "causal_reacceleration_context_audit_v1"

FROZEN_PHASE7K2_BUILDER = "1.0.0"
FROZEN_PHASE7K4_BUILDER = "1.0.0"
FROZEN_PHASE7L1_BUILDER = "1.0.1"
FROZEN_POLICY = "TEST_COOLDOWN_60S"

EXPECTED_CANDIDATES = 4_037
PRIMARY_OBSERVATION_HORIZON_SEC = 3
PRIOR50_LOOKBACK_SEC = 1_800

CONTROL_BINS = 3
BOOK_BINS = 3
MIN_CELL_SIDE_N = 10
PRIMARY_FEASIBILITY_N = 250
DIRECTION_FEASIBILITY_N = 125
MIN_CORE_FEATURES_PASS = 3

PRIMARY_MOMENTUM_CONTROL = "directional_mid_move_bps"
PRIMARY_AGGRESSION_CONTROL = "directional_aggressive_quote"

# sign orients HIGH toward the previously hypothesized continuation-favoring
# liquidity-failure state. No current-candidate outcome is used here.
CORE_BOOK_FEATURES: tuple[tuple[str, int], ...] = (
    ("opposing__durable_unrecovered_loss_share_initial", +1),
    ("opposing__peak_initial_queue_loss_share", +1),
    ("opposing__retained_initial_quote_ratio", -1),
    ("support_minus_opposing_retained_ratio", +1),
)

DEFINITIONS = (
    "REARM_STATE",
    "REARM_IMMEDIATE_PRIOR50",
    "REARM_ANY_PRIOR50_30M",
    "REARM_ANY_PRIOR50_30M_NO_OPPOSITE50",
)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_K2_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "causal_rearm_timing_audit_v1"
)
DEFAULT_K4_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "candidate_redundancy_earliest_availability_audit_v1"
)
DEFAULT_L1_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "repeatable_candidate_participant_interaction_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" / "perps_causal_reacceleration_context_audit"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_time(sec: Any) -> str | None:
    try:
        x = float(sec)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return datetime.fromtimestamp(int(x), tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def sha256_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(x) for x in ids)).encode("utf-8")
    ).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l3-", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l3-", suffix=".csv", dir=str(path.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Phase 7L.3 Parquet output requires pyarrow in the project .venv."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l3-", suffix=".parquet", dir=str(path.parent)
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
    if str(meta.get("status")) != "PASS":
        raise RuntimeError(
            f"{label} must be PASS; found status={meta.get('status')!r}"
        )
    return meta


def percentile_bin(s: pd.Series, n_bins: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    mask = x.notna()
    if not mask.any():
        return out
    ranks = x.loc[mask].rank(method="first", pct=True)
    out.loc[mask] = (
        np.ceil(ranks * int(n_bins)).clip(1, n_bins)
        .astype(int).astype("Int64")
    )
    return out


def load_inputs(
    k2_root: Path,
    k4_root: Path,
    l1_root: Path,
):
    k2_meta = load_metadata(
        k2_root, FROZEN_PHASE7K2_BUILDER, "Phase 7K.2"
    )
    k4_meta = load_metadata(
        k4_root, FROZEN_PHASE7K4_BUILDER, "Phase 7K.4"
    )
    l1_meta = load_metadata(
        l1_root, FROZEN_PHASE7L1_BUILDER, "Phase 7L.1"
    )

    cpath = k2_root / "candidates.parquet"
    opath = k2_root / "candidate_outcomes.parquet"
    apath = k4_root / "candidate_onset_associations.parquet"
    mpath = l1_root / "candidate_mechanics.parquet"

    for p in (cpath, opath, apath, mpath):
        if not p.exists():
            raise RuntimeError(f"Missing required frozen input: {p}")

    candidates = pd.read_parquet(cpath)
    outcomes = pd.read_parquet(opath)
    associations = pd.read_parquet(apath)
    mechanics = pd.read_parquet(mpath)

    candidates = candidates[
        candidates["policy_id"].astype(str) == FROZEN_POLICY
    ].copy()
    outcomes = outcomes[
        outcomes["policy_id"].astype(str) == FROZEN_POLICY
    ].copy()
    associations = associations[
        associations["policy_id"].astype(str) == FROZEN_POLICY
    ].copy()

    return (
        k2_meta, k4_meta, l1_meta,
        candidates, outcomes, associations, mechanics
    )


def validate_k2_contract(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    require_columns(
        candidates,
        [
            "candidate_id", "policy_id", "candidate_date",
            "direction", "direction_label",
            "origin_kind", "origin_second", "release_second",
            "candidate_second",
            "prior_same_direction_candidate_second",
            "same_direction_candidate_ordinal",
            "reset_threshold_bps", "release_threshold_bps",
            "candidate_threshold_bps", "cooldown_sec",
        ],
        "Phase-7K.2 candidates",
    )
    require_columns(
        outcomes,
        [
            "candidate_id", "policy_id", "direction", "candidate_second",
            "seconds_to_50_from_candidate",
        ],
        "Phase-7K.2 candidate outcomes",
    )

    if len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} frozen 60s candidates; "
            f"found {len(candidates):,}"
        )
    if candidates["candidate_id"].duplicated().any():
        raise RuntimeError("Frozen 60s candidates contain duplicate candidate_id")

    ids = set(candidates["candidate_id"].astype(str))
    out_ids = set(outcomes["candidate_id"].astype(str))
    if ids != out_ids:
        raise RuntimeError(
            "Selected Phase-7K.2 outcome IDs do not match candidate IDs: "
            f"missing_outcomes={len(ids-out_ids)} extra_outcomes={len(out_ids-ids)}"
        )

    contracts = {
        "reset": sorted(
            pd.to_numeric(candidates["reset_threshold_bps"], errors="coerce")
            .dropna().unique().tolist()
        ),
        "release": sorted(
            pd.to_numeric(candidates["release_threshold_bps"], errors="coerce")
            .dropna().unique().tolist()
        ),
        "candidate": sorted(
            pd.to_numeric(candidates["candidate_threshold_bps"], errors="coerce")
            .dropna().unique().tolist()
        ),
        "cooldown": sorted(
            pd.to_numeric(candidates["cooldown_sec"], errors="coerce")
            .dropna().unique().tolist()
        ),
    }
    expected = {
        "reset": [5.0],
        "release": [10.0],
        "candidate": [35.0],
        "cooldown": [60],
    }
    if contracts != expected:
        raise RuntimeError(
            f"Frozen candidate contract mismatch: observed={contracts} expected={expected}"
        )


def build_prior50_ledger(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert old-candidate outcome records into historical 50-bp confirmation
    events. Later classification may only use a hit if hit_second <= the CURRENT
    candidate's causal origin second.
    """
    c = candidates[
        ["candidate_id", "direction", "candidate_second"]
    ].copy()
    c["candidate_id"] = c["candidate_id"].astype(str)
    o = outcomes[
        ["candidate_id", "seconds_to_50_from_candidate"]
    ].copy()
    o["candidate_id"] = o["candidate_id"].astype(str)

    x = c.merge(o, on="candidate_id", how="left", validate="one_to_one")
    seconds = pd.to_numeric(
        x["seconds_to_50_from_candidate"], errors="coerce"
    )
    csec = pd.to_numeric(x["candidate_second"], errors="coerce")
    x["hit_50_second"] = csec + seconds
    x = x[
        x["hit_50_second"].notna()
        & (x["hit_50_second"] >= csec)
    ].copy()
    x["hit_50_second"] = x["hit_50_second"].astype("int64")
    return x.sort_values(
        ["hit_50_second", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def prior_hit_indexes(ledger: pd.DataFrame) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for direction in (+1, -1):
        g = ledger[
            pd.to_numeric(ledger["direction"], errors="coerce") == direction
        ].sort_values("hit_50_second", kind="mergesort")
        out[direction] = {
            "hit_seconds": g["hit_50_second"].astype(int).tolist(),
            "candidate_ids": g["candidate_id"].astype(str).tolist(),
        }
    return out


def latest_hit_before(
    index: dict[int, dict[str, Any]],
    direction: int,
    second: int,
    *,
    min_second: int | None = None,
) -> tuple[int | None, str | None]:
    d = index.get(int(direction), {"hit_seconds": [], "candidate_ids": []})
    secs = d["hit_seconds"]
    ids = d["candidate_ids"]
    j = bisect_right(secs, int(second)) - 1
    if j < 0:
        return None, None
    hit = int(secs[j])
    if min_second is not None and hit < int(min_second):
        return None, None
    return hit, str(ids[j])


def has_hit_between(
    index: dict[int, dict[str, Any]],
    direction: int,
    lo_exclusive: int,
    hi_inclusive: int,
) -> bool:
    secs = index.get(
        int(direction), {"hit_seconds": []}
    )["hit_seconds"]
    j = bisect_right(secs, int(lo_exclusive))
    return j < len(secs) and int(secs[j]) <= int(hi_inclusive)


def build_causal_context_labels(
    candidates: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    c = candidates.copy()
    c["candidate_id"] = c["candidate_id"].astype(str)
    c = c.sort_values(
        ["candidate_second", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)

    hit_index = prior_hit_indexes(ledger)
    by_dir_second = {
        (int(r.direction), int(r.candidate_second)): str(r.candidate_id)
        for r in c[
            ["candidate_id", "direction", "candidate_second"]
        ].itertuples(index=False)
    }
    hit_by_candidate = dict(
        zip(
            ledger["candidate_id"].astype(str),
            ledger["hit_50_second"].astype(int),
        )
    )

    rows = []
    for r in c.to_dict("records"):
        cid = str(r["candidate_id"])
        direction = int(r["direction"])
        origin = int(r["origin_second"])
        release = int(r["release_second"])
        csec = int(r["candidate_second"])
        origin_kind = str(r.get("origin_kind") or "")

        rearm_state = origin_kind == "REARM_RESET"

        prior_sec_raw = safe_float(
            r.get("prior_same_direction_candidate_second")
        )
        immediate_prior_sec = (
            int(prior_sec_raw)
            if math.isfinite(prior_sec_raw)
            else None
        )
        immediate_prior_id = (
            by_dir_second.get((direction, immediate_prior_sec))
            if immediate_prior_sec is not None
            else None
        )
        immediate_prior_hit50 = (
            hit_by_candidate.get(str(immediate_prior_id))
            if immediate_prior_id is not None
            else None
        )
        immediate_prior50_before_origin = bool(
            immediate_prior_hit50 is not None
            and int(immediate_prior_hit50) <= origin
        )

        lookback_start = origin - PRIOR50_LOOKBACK_SEC
        latest_same_hit, latest_same_cid = latest_hit_before(
            hit_index,
            direction,
            origin,
            min_second=lookback_start,
        )
        any_prior50_30m = latest_same_hit is not None
        opposite50_since = bool(
            latest_same_hit is not None
            and has_hit_between(
                hit_index,
                -direction,
                int(latest_same_hit),
                origin,
            )
        )

        defs = {
            "REARM_STATE": rearm_state,
            "REARM_IMMEDIATE_PRIOR50": (
                rearm_state and immediate_prior50_before_origin
            ),
            "REARM_ANY_PRIOR50_30M": (
                rearm_state and any_prior50_30m
            ),
            "REARM_ANY_PRIOR50_30M_NO_OPPOSITE50": (
                rearm_state
                and any_prior50_30m
                and not opposite50_since
            ),
        }

        row = {
            "candidate_id": cid,
            "candidate_date": str(r["candidate_date"]),
            "direction": direction,
            "direction_label": str(r["direction_label"]),
            "candidate_second": csec,
            "candidate_time": iso_time(csec),
            "origin_second": origin,
            "origin_time": iso_time(origin),
            "release_second": release,
            "origin_kind": origin_kind,
            "same_direction_candidate_ordinal": int(
                r["same_direction_candidate_ordinal"]
            ),
            "prior_same_direction_candidate_second": (
                immediate_prior_sec
                if immediate_prior_sec is not None
                else math.nan
            ),
            "immediate_prior_candidate_id": immediate_prior_id,
            "immediate_prior_50_hit_second": (
                int(immediate_prior_hit50)
                if immediate_prior_hit50 is not None
                else math.nan
            ),
            "immediate_prior_50_confirmed_before_origin": (
                immediate_prior50_before_origin
            ),
            "latest_same_direction_50_hit_30m_second": (
                int(latest_same_hit)
                if latest_same_hit is not None
                else math.nan
            ),
            "latest_same_direction_50_hit_30m_candidate_id": latest_same_cid,
            "seconds_since_latest_same_direction_50_hit": (
                int(origin - latest_same_hit)
                if latest_same_hit is not None
                else math.nan
            ),
            "opposite_50_confirmed_since_latest_same_50": opposite50_since,
        }
        for definition, flag in defs.items():
            row[f"is_reaccel__{definition}"] = bool(flag)
        rows.append(row)

    return pd.DataFrame(rows)


def build_structural_roles(
    associations: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        associations,
        [
            "candidate_id", "impulse_id",
            "impulse_ordinal_in_physical_leg",
            "is_reacceleration",
            "candidate_rank_in_onset_window",
        ],
        "Phase-7K.4 candidate_onset_associations",
    )
    a = associations.copy()
    a["candidate_id"] = a["candidate_id"].astype(str)

    rows = []
    for cid, g in a.groupby("candidate_id", sort=False):
        impulses = sorted(set(g["impulse_id"].astype(str)))
        roles = set(
            "REACCEL" if bool(x) else "FIRST"
            for x in g["is_reacceleration"].fillna(False).astype(bool)
        )
        if len(impulses) == 1 and len(roles) == 1:
            r = g.sort_values(
                "candidate_rank_in_onset_window", kind="mergesort"
            ).iloc[0]
            rows.append({
                "candidate_id": cid,
                "structural_role": next(iter(roles)),
                "structural_impulse_id": str(r["impulse_id"]),
                "structural_ordinal": int(
                    r["impulse_ordinal_in_physical_leg"]
                ),
                "structural_role_confident": True,
            })
        else:
            rows.append({
                "candidate_id": cid,
                "structural_role": "AMBIGUOUS",
                "structural_impulse_id": None,
                "structural_ordinal": math.nan,
                "structural_role_confident": False,
            })
    return pd.DataFrame(rows)


def context_counts(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in DEFINITIONS:
        flag = bool_series(labels[f"is_reaccel__{definition}"])
        for segment, mask in (
            ("ALL", pd.Series(True, index=labels.index)),
            ("UP", pd.to_numeric(labels["direction"], errors="coerce") > 0),
            ("DOWN", pd.to_numeric(labels["direction"], errors="coerce") < 0),
        ):
            x = labels.loc[mask]
            f = flag.loc[mask]
            rows.append({
                "definition": definition,
                "segment": segment,
                "candidates": int(len(x)),
                "reaccel_context_candidates": int(f.sum()),
                "non_reaccel_context_candidates": int((~f).sum()),
                "reaccel_context_share": (
                    float(f.mean()) if len(f) else math.nan
                ),
            })
    return pd.DataFrame(rows)


def structural_agreement(
    labels: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    x = labels.merge(
        roles,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    confident = x[
        x["structural_role_confident"].fillna(False).astype(bool)
        & x["structural_role"].isin(["FIRST", "REACCEL"])
    ].copy()

    rows = []
    actual_re = confident["structural_role"] == "REACCEL"
    actual_first = confident["structural_role"] == "FIRST"
    n_re = int(actual_re.sum())
    n_first = int(actual_first.sum())

    for definition in DEFINITIONS:
        pred = bool_series(
            confident[f"is_reaccel__{definition}"]
        )
        tp = int((pred & actual_re).sum())
        fn = int((~pred & actual_re).sum())
        fp = int((pred & actual_first).sum())
        tn = int((~pred & actual_first).sum())

        recall = tp / n_re if n_re else math.nan
        specificity = tn / n_first if n_first else math.nan
        fpr = fp / n_first if n_first else math.nan
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        agreement = (
            (tp + tn) / len(confident)
            if len(confident) else math.nan
        )
        bal = (
            0.5 * (recall + specificity)
            if math.isfinite(recall) and math.isfinite(specificity)
            else math.nan
        )

        rows.append({
            "definition": definition,
            "confident_structural_candidates": int(len(confident)),
            "structural_reaccel_candidates": n_re,
            "structural_first_candidates": n_first,
            "true_positive_reaccel": tp,
            "false_negative_reaccel": fn,
            "false_positive_first_as_reaccel": fp,
            "true_negative_first": tn,
            "reaccel_recall": recall,
            "first_specificity": specificity,
            "first_false_positive_rate": fpr,
            "precision_among_structurally_labeled": precision,
            "overall_agreement": agreement,
            "balanced_accuracy": bal,
            "note": (
                "UNASSOCIATED candidates excluded; precision is only among "
                "confidently structurally-labelled candidates."
            ),
        })
    return pd.DataFrame(rows)


def prepare_mechanics_for_feasibility(
    mechanics: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "candidate_id", "candidate_date", "direction",
        "observation_horizon_sec", "window_valid",
        PRIMARY_MOMENTUM_CONTROL, PRIMARY_AGGRESSION_CONTROL,
    ] + [x[0] for x in CORE_BOOK_FEATURES]
    require_columns(mechanics, required, "Phase-7L.1 candidate_mechanics")

    d = mechanics[
        pd.to_numeric(
            mechanics["observation_horizon_sec"], errors="coerce"
        ) == PRIMARY_OBSERVATION_HORIZON_SEC
    ].copy()
    d["candidate_id"] = d["candidate_id"].astype(str)
    if d["candidate_id"].duplicated().any():
        raise RuntimeError(
            "Phase-7L.1 primary-horizon mechanics contain duplicate candidate IDs"
        )

    d["momentum_tercile"] = percentile_bin(
        pd.to_numeric(d[PRIMARY_MOMENTUM_CONTROL], errors="coerce"),
        CONTROL_BINS,
    ).astype(float)
    d["aggression_tercile"] = percentile_bin(
        np.log1p(
            pd.to_numeric(
                d[PRIMARY_AGGRESSION_CONTROL], errors="coerce"
            ).clip(lower=0.0)
        ),
        CONTROL_BINS,
    ).astype(float)

    for feature, sign in CORE_BOOK_FEATURES:
        oriented = (
            pd.to_numeric(d[feature], errors="coerce") * float(sign)
        )
        d[f"book_tercile__{feature}"] = percentile_bin(
            oriented,
            BOOK_BINS,
        ).astype(float)

    return d


def feasibility_for_definition(
    mechanics: pd.DataFrame,
    labels: pd.DataFrame,
    definition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = [
        "candidate_id", "direction",
        f"is_reaccel__{definition}",
    ]
    x = mechanics.merge(
        labels[label_cols],
        on=["candidate_id", "direction"],
        how="left",
        validate="one_to_one",
    )
    x = x[
        bool_series(x[f"is_reaccel__{definition}"])
        & bool_series(x["window_valid"])
    ].copy()

    summary_rows = []
    cell_rows = []

    segments = {
        "ALL": pd.Series(True, index=x.index),
        "UP": pd.to_numeric(x["direction"], errors="coerce") > 0,
        "DOWN": pd.to_numeric(x["direction"], errors="coerce") < 0,
    }

    for segment, segment_mask in segments.items():
        d = x.loc[segment_mask].copy()
        threshold = (
            PRIMARY_FEASIBILITY_N
            if segment == "ALL"
            else DIRECTION_FEASIBILITY_N
        )
        for feature, sign in CORE_BOOK_FEATURES:
            bcol = f"book_tercile__{feature}"
            usable = d[
                pd.to_numeric(d["momentum_tercile"], errors="coerce").notna()
                & pd.to_numeric(d["aggression_tercile"], errors="coerce").notna()
                & pd.to_numeric(d[bcol], errors="coerce").isin([1, BOOK_BINS])
            ].copy()

            total_common = 0
            cells_used = 0
            total_hi = 0
            total_lo = 0

            for (m, a), g in usable.groupby(
                ["momentum_tercile", "aggression_tercile"],
                dropna=False,
            ):
                hi = int(
                    (
                        pd.to_numeric(g[bcol], errors="coerce")
                        == BOOK_BINS
                    ).sum()
                )
                lo = int(
                    (
                        pd.to_numeric(g[bcol], errors="coerce")
                        == 1
                    ).sum()
                )
                cell_common = (
                    2 * min(hi, lo)
                    if hi >= MIN_CELL_SIDE_N and lo >= MIN_CELL_SIDE_N
                    else 0
                )
                if cell_common:
                    cells_used += 1
                    total_common += cell_common
                    total_hi += hi
                    total_lo += lo

                cell_rows.append({
                    "definition": definition,
                    "segment": segment,
                    "feature": feature,
                    "momentum_tercile": int(m),
                    "aggression_tercile": int(a),
                    "high_book_n": hi,
                    "low_book_n": lo,
                    "cell_usable": bool(cell_common > 0),
                    "cell_common_support_n": int(cell_common),
                    "min_cell_side_n": MIN_CELL_SIDE_N,
                })

            summary_rows.append({
                "definition": definition,
                "segment": segment,
                "feature": feature,
                "reaccel_candidates": int(len(d)),
                "usable_high_low_rows": int(len(usable)),
                "cells_used": int(cells_used),
                "high_n_in_used_cells": int(total_hi),
                "low_n_in_used_cells": int(total_lo),
                "common_support_n": int(total_common),
                "feasibility_threshold_n": int(threshold),
                "feature_feasible": bool(total_common >= threshold),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(cell_rows)


def aggregate_feasibility(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in DEFINITIONS:
        for segment in ("ALL", "UP", "DOWN"):
            x = summary[
                (summary["definition"] == definition)
                & (summary["segment"] == segment)
            ]
            pass_n = int(x["feature_feasible"].astype(bool).sum())
            rows.append({
                "definition": definition,
                "segment": segment,
                "core_features": int(len(x)),
                "core_features_feasible": pass_n,
                "min_core_features_required": MIN_CORE_FEATURES_PASS,
                "segment_feasible": bool(
                    pass_n >= MIN_CORE_FEATURES_PASS
                ),
                "minimum_common_support_n": (
                    int(x["common_support_n"].min()) if len(x) else 0
                ),
                "median_common_support_n": (
                    float(x["common_support_n"].median())
                    if len(x) else math.nan
                ),
                "maximum_common_support_n": (
                    int(x["common_support_n"].max()) if len(x) else 0
                ),
            })
    return pd.DataFrame(rows)


def timing_profile(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in DEFINITIONS:
        x = labels[
            bool_series(labels[f"is_reaccel__{definition}"])
        ].copy()
        ages = pd.to_numeric(
            x["seconds_since_latest_same_direction_50_hit"],
            errors="coerce",
        ).dropna()
        rows.append({
            "definition": definition,
            "reaccel_candidates": int(len(x)),
            "with_prior50_age": int(len(ages)),
            "prior50_age_median_sec": (
                float(ages.median()) if len(ages) else math.nan
            ),
            "prior50_age_p25_sec": (
                float(ages.quantile(0.25)) if len(ages) else math.nan
            ),
            "prior50_age_p75_sec": (
                float(ages.quantile(0.75)) if len(ages) else math.nan
            ),
            "prior50_age_p90_sec": (
                float(ages.quantile(0.90)) if len(ages) else math.nan
            ),
        })
    return pd.DataFrame(rows)


def build_invariants(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    ledger: pd.DataFrame,
    labels: pd.DataFrame,
    mechanics3: pd.DataFrame,
    structural: pd.DataFrame,
    k2_meta: dict[str, Any],
    l1_meta: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({
            "invariant": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    add(
        "candidate_count",
        len(candidates) == EXPECTED_CANDIDATES,
        len(candidates),
        EXPECTED_CANDIDATES,
    )
    add(
        "candidate_ids_unique",
        not candidates["candidate_id"].duplicated().any(),
        int(candidates["candidate_id"].duplicated().sum()),
        0,
    )
    add(
        "labels_one_to_one",
        len(labels) == len(candidates)
        and not labels["candidate_id"].duplicated().any(),
        len(labels),
        len(candidates),
    )
    add(
        "outcomes_one_to_one",
        len(outcomes) == len(candidates)
        and not outcomes["candidate_id"].duplicated().any(),
        len(outcomes),
        len(candidates),
    )
    add(
        "primary_mechanics_one_per_candidate",
        len(mechanics3) == EXPECTED_CANDIDATES
        and not mechanics3["candidate_id"].duplicated().any(),
        len(mechanics3),
        EXPECTED_CANDIDATES,
    )
    add(
        "candidate_hash_matches_phase7l1",
        sha256_ids(candidates["candidate_id"].astype(str))
        == str(l1_meta.get("candidate_id_set_sha256")),
        sha256_ids(candidates["candidate_id"].astype(str)),
        l1_meta.get("candidate_id_set_sha256"),
    )

    bad_ledger = 0
    if len(ledger):
        bad_ledger = int(
            (
                pd.to_numeric(ledger["hit_50_second"], errors="coerce")
                < pd.to_numeric(
                    ledger["candidate_second"], errors="coerce"
                )
            ).sum()
        )
    add(
        "prior50_ledger_hits_not_before_source_candidate",
        bad_ledger == 0,
        bad_ledger,
        0,
    )

    bad_current_leak = 0
    hit_by_id = dict(
        zip(
            ledger["candidate_id"].astype(str),
            ledger["hit_50_second"].astype(int),
        )
    )
    # A current candidate's own future 50 hit must never be the historical
    # evidence used to classify that same candidate.
    for r in labels.to_dict("records"):
        cid = str(r["candidate_id"])
        current_hit = hit_by_id.get(cid)
        latest_prior = safe_float(
            r.get("latest_same_direction_50_hit_30m_second")
        )
        if (
            current_hit is not None
            and math.isfinite(latest_prior)
            and int(latest_prior) == int(current_hit)
        ):
            bad_current_leak += 1
    add(
        "no_current_candidate_50_used_as_its_own_context",
        bad_current_leak == 0,
        bad_current_leak,
        0,
    )

    causal_order_bad = int(
        (
            pd.to_numeric(labels["origin_second"], errors="coerce")
            > pd.to_numeric(labels["release_second"], errors="coerce")
        ).sum()
        + (
            pd.to_numeric(labels["release_second"], errors="coerce")
            > pd.to_numeric(labels["candidate_second"], errors="coerce")
        ).sum()
    )
    add(
        "origin_release_candidate_order",
        causal_order_bad == 0,
        causal_order_bad,
        0,
    )

    impossible_immediate = int(
        (
            bool_series(
                labels[
                    "immediate_prior_50_confirmed_before_origin"
                ]
            )
            & (
                pd.to_numeric(
                    labels["immediate_prior_50_hit_second"],
                    errors="coerce",
                )
                > pd.to_numeric(labels["origin_second"], errors="coerce")
            )
        ).sum()
    )
    add(
        "immediate_prior50_is_historical_by_current_origin",
        impossible_immediate == 0,
        impossible_immediate,
        0,
    )

    for definition in DEFINITIONS[1:]:
        child = bool_series(labels[f"is_reaccel__{definition}"])
        parent = bool_series(labels["is_reaccel__REARM_STATE"])
        violations = int((child & ~parent).sum())
        add(
            f"{definition}_subset_of_rearm_state",
            violations == 0,
            violations,
            0,
        )

    confident_roles = int(
        structural.get(
            "structural_role_confident",
            pd.Series(dtype=bool),
        ).fillna(False).astype(bool).sum()
    )
    add(
        "structural_audit_has_confident_labels",
        confident_roles > 0,
        confident_roles,
        ">0",
    )

    add(
        "phase7k2_policy_hash_present",
        FROZEN_POLICY in (
            k2_meta.get("candidate_id_set_sha256_by_policy", {}) or {}
        ),
        FROZEN_POLICY in (
            k2_meta.get("candidate_id_set_sha256_by_policy", {}) or {}
        ),
        True,
    )

    return pd.DataFrame(rows)


def ranked_review_table(
    agreement: pd.DataFrame,
    feasibility_agg: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    a = agreement.copy()
    f = feasibility_agg[
        feasibility_agg["segment"] == "ALL"
    ][
        [
            "definition", "segment_feasible",
            "core_features_feasible",
            "minimum_common_support_n",
            "median_common_support_n",
        ]
    ]
    c = counts[counts["segment"] == "ALL"][
        ["definition", "reaccel_context_candidates"]
    ]
    x = a.merge(f, on="definition", how="left").merge(
        c, on="definition", how="left"
    )

    # Ranking uses ONLY structural agreement and feasibility. No 75/100 rate.
    x["review_eligible"] = x["segment_feasible"].fillna(False)
    x["review_rank_score"] = np.where(
        x["review_eligible"],
        pd.to_numeric(x["balanced_accuracy"], errors="coerce"),
        -1.0,
    )
    return x.sort_values(
        [
            "review_eligible", "review_rank_score",
            "reaccel_context_candidates",
        ],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def findings_text(
    status: str,
    counts: pd.DataFrame,
    agreement: pd.DataFrame,
    feasibility_agg: pd.DataFrame,
    ranked: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7L.3 — Causal re-acceleration context + feasibility audit",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{status}**",
        f"- Frozen policy: **{FROZEN_POLICY}**",
        f"- Candidates: **{EXPECTED_CANDIDATES:,}**",
        "- Current-candidate 75/100 outcomes used to select definition: **NO**",
        "- Raw depth/trade replay: **NO**",
        "",
        "## Plain-English purpose",
        "",
        "Find a causal label for 'this 35-bp candidate is a repeat/re-acceleration attempt'",
        "that includes both later winners and later failures, then verify BEFORE another analysis",
        "that enough observations survive momentum x aggression x book-state conditioning.",
        "",
        "## Candidate counts",
        "",
    ]

    cc = counts[counts["segment"] == "ALL"]
    for _, r in cc.iterrows():
        lines.append(
            f"- {r['definition']}: "
            f"{int(r['reaccel_context_candidates']):,}/"
            f"{int(r['candidates']):,} "
            f"({100*safe_float(r['reaccel_context_share']):.1f}%)."
        )

    lines += [
        "",
        "## Structural agreement on confidently-labelled Phase-7K.4 cases",
        "",
        "UNASSOCIATED candidates are not counted as negatives.",
        "",
    ]
    for _, r in agreement.sort_values(
        "balanced_accuracy", ascending=False
    ).iterrows():
        lines.append(
            f"- {r['definition']}: "
            f"reaccel recall={100*safe_float(r['reaccel_recall']):.1f}%, "
            f"FIRST specificity={100*safe_float(r['first_specificity']):.1f}%, "
            f"balanced accuracy={100*safe_float(r['balanced_accuracy']):.1f}%, "
            f"precision among labelled={100*safe_float(r['precision_among_structurally_labeled']):.1f}%."
        )

    lines += [
        "",
        "## Pre-compute feasibility",
        "",
        f"Primary gate: >= {PRIMARY_FEASIBILITY_N} common-support observations for "
        f"at least {MIN_CORE_FEATURES_PASS}/{len(CORE_BOOK_FEATURES)} core book features.",
        "",
    ]
    ff = feasibility_agg[feasibility_agg["segment"] == "ALL"]
    for _, r in ff.iterrows():
        lines.append(
            f"- {r['definition']}: "
            f"{int(r['core_features_feasible'])}/{int(r['core_features'])} "
            f"core features feasible; "
            f"min common-support N={int(r['minimum_common_support_n'])}; "
            f"median={safe_float(r['median_common_support_n']):.0f}; "
            f"segment_feasible={bool(r['segment_feasible'])}."
        )

    lines += [
        "",
        "## Review ranking",
        "",
        "This ranking uses structural agreement + feasibility only. It does NOT use future",
        "75/100 performance and does not automatically freeze a definition.",
        "",
    ]
    for i, r in ranked.iterrows():
        lines.append(
            f"{i+1}. {r['definition']} — "
            f"eligible={bool(r['review_eligible'])}; "
            f"reaccel N={int(r['reaccel_context_candidates'])}; "
            f"balanced accuracy={100*safe_float(r['balanced_accuracy']):.1f}%; "
            f"core feasible={int(r['core_features_feasible'])}/{len(CORE_BOOK_FEATURES)}."
        )

    lines += [
        "",
        "## Decision discipline",
        "",
        "- Do not choose the definition from 75/100 continuation rates.",
        "- First choose a causal context definition that has acceptable structural meaning",
        "  and enough common support.",
        "- Only after review/freeze should the already-built 7L.1 mechanics be used for",
        "  a dedicated re-acceleration continuation analysis.",
        "- No new depth replay is needed for that downstream analysis.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> int:
    # Four causal candidates. c2 and c3 are REARM_RESET. c1 reaches 50 before
    # c2 origin. Opposite d1 reaches 50 between c2/c3 contexts.
    candidates = pd.DataFrame([
        {
            "candidate_id": "c1", "policy_id": FROZEN_POLICY,
            "candidate_date": "2026-06-01", "direction": 1,
            "direction_label": "UP", "origin_kind": "INITIAL_EXTREME",
            "origin_second": 100, "release_second": 110,
            "candidate_second": 130,
            "prior_same_direction_candidate_second": math.nan,
            "same_direction_candidate_ordinal": 1,
            "reset_threshold_bps": 5.0, "release_threshold_bps": 10.0,
            "candidate_threshold_bps": 35.0, "cooldown_sec": 60,
        },
        {
            "candidate_id": "c2", "policy_id": FROZEN_POLICY,
            "candidate_date": "2026-06-01", "direction": 1,
            "direction_label": "UP", "origin_kind": "REARM_RESET",
            "origin_second": 220, "release_second": 230,
            "candidate_second": 250,
            "prior_same_direction_candidate_second": 130,
            "same_direction_candidate_ordinal": 2,
            "reset_threshold_bps": 5.0, "release_threshold_bps": 10.0,
            "candidate_threshold_bps": 35.0, "cooldown_sec": 60,
        },
        {
            "candidate_id": "d1", "policy_id": FROZEN_POLICY,
            "candidate_date": "2026-06-01", "direction": -1,
            "direction_label": "DOWN", "origin_kind": "INITIAL_EXTREME",
            "origin_second": 260, "release_second": 270,
            "candidate_second": 290,
            "prior_same_direction_candidate_second": math.nan,
            "same_direction_candidate_ordinal": 1,
            "reset_threshold_bps": 5.0, "release_threshold_bps": 10.0,
            "candidate_threshold_bps": 35.0, "cooldown_sec": 60,
        },
        {
            "candidate_id": "c3", "policy_id": FROZEN_POLICY,
            "candidate_date": "2026-06-01", "direction": 1,
            "direction_label": "UP", "origin_kind": "REARM_RESET",
            "origin_second": 380, "release_second": 390,
            "candidate_second": 410,
            "prior_same_direction_candidate_second": 250,
            "same_direction_candidate_ordinal": 3,
            "reset_threshold_bps": 5.0, "release_threshold_bps": 10.0,
            "candidate_threshold_bps": 35.0, "cooldown_sec": 60,
        },
    ])
    outcomes = pd.DataFrame([
        {
            "candidate_id": "c1", "policy_id": FROZEN_POLICY,
            "direction": 1, "candidate_second": 130,
            "seconds_to_50_from_candidate": 40,
        },
        {
            "candidate_id": "c2", "policy_id": FROZEN_POLICY,
            "direction": 1, "candidate_second": 250,
            "seconds_to_50_from_candidate": math.nan,
        },
        {
            "candidate_id": "d1", "policy_id": FROZEN_POLICY,
            "direction": -1, "candidate_second": 290,
            "seconds_to_50_from_candidate": 30,
        },
        {
            "candidate_id": "c3", "policy_id": FROZEN_POLICY,
            "direction": 1, "candidate_second": 410,
            "seconds_to_50_from_candidate": math.nan,
        },
    ])

    ledger = build_prior50_ledger(candidates, outcomes)
    labels = build_causal_context_labels(candidates, ledger)
    l = labels.set_index("candidate_id")

    assert bool(l.loc["c2", "is_reaccel__REARM_STATE"])
    assert bool(l.loc["c2", "is_reaccel__REARM_IMMEDIATE_PRIOR50"])
    assert bool(l.loc["c2", "is_reaccel__REARM_ANY_PRIOR50_30M"])
    assert bool(
        l.loc[
            "c2",
            "is_reaccel__REARM_ANY_PRIOR50_30M_NO_OPPOSITE50",
        ]
    )

    assert bool(l.loc["c3", "is_reaccel__REARM_STATE"])
    assert not bool(
        l.loc["c3", "is_reaccel__REARM_IMMEDIATE_PRIOR50"]
    )
    assert bool(
        l.loc["c3", "is_reaccel__REARM_ANY_PRIOR50_30M"]
    )
    assert not bool(
        l.loc[
            "c3",
            "is_reaccel__REARM_ANY_PRIOR50_30M_NO_OPPOSITE50",
        ]
    )

    # Current candidate c1's own hit at 170 cannot classify itself because
    # its current origin is 100.
    assert math.isnan(
        safe_float(
            l.loc[
                "c1",
                "latest_same_direction_50_hit_30m_second",
            ]
        )
    )

    # Synthetic feasibility: 9 control cells, enough high/low observations.
    rows = []
    for m in (1, 2, 3):
        for a in (1, 2, 3):
            for side in (1, 3):
                for j in range(12):
                    row = {
                        "candidate_id": f"x-{m}-{a}-{side}-{j}",
                        "candidate_date": "2026-06-01",
                        "direction": 1,
                        "observation_horizon_sec": 3,
                        "window_valid": True,
                        "momentum_tercile": float(m),
                        "aggression_tercile": float(a),
                    }
                    for feature, _ in CORE_BOOK_FEATURES:
                        row[f"book_tercile__{feature}"] = float(side)
                    rows.append(row)
    mech = pd.DataFrame(rows)
    lab = pd.DataFrame({
        "candidate_id": mech["candidate_id"],
        "direction": 1,
        "is_reaccel__REARM_STATE": True,
    })
    fs, cells = feasibility_for_definition(
        mech, lab, "REARM_STATE"
    )
    all_rows = fs[fs["segment"] == "ALL"]
    assert (all_rows["common_support_n"] == 216).all()
    assert (all_rows["cells_used"] == 9).all()

    print("SELF-TEST PASS")
    print(
        "Validated causal REARM_RESET classification, historical prior-50 use, "
        "opposite-50 interruption logic, no current-candidate self-leakage, "
        "and pre-compute common-support feasibility accounting."
    )
    return 0


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Phase 7L.3 causal re-acceleration context + feasibility audit"
    )
    p.add_argument("--phase7k2-root", type=Path, default=DEFAULT_K2_ROOT)
    p.add_argument("--phase7k4-root", type=Path, default=DEFAULT_K4_ROOT)
    p.add_argument("--phase7l1-root", type=Path, default=DEFAULT_L1_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    print("Building Phase 7L.3 causal re-acceleration context audit...")
    print(f"Builder version        : {BUILDER_VERSION}")
    print(f"Frozen candidate policy: {FROZEN_POLICY}")
    print("Definitions            :")
    for d in DEFINITIONS:
        print(f"  - {d}")
    print("Current 75/100 outcomes: NOT USED")
    print("Raw depth replay       : NONE")
    print("Raw trade replay       : NONE")
    print("Primary purpose        : causal context + feasibility BEFORE subgroup analysis")

    print("[1/7] Loading frozen inputs...")
    (
        k2_meta, k4_meta, l1_meta,
        candidates, outcomes, associations, mechanics
    ) = load_inputs(
        args.phase7k2_root,
        args.phase7k4_root,
        args.phase7l1_root,
    )
    validate_k2_contract(candidates, outcomes)
    print(
        f"      candidates={len(candidates):,} "
        f"mechanics={len(mechanics):,} "
        f"K4 onset associations={len(associations):,}"
    )

    print("[2/7] Building historical prior-50 event ledger...")
    ledger = build_prior50_ledger(candidates, outcomes)
    print(f"      prior candidate 50-confirmations stored={len(ledger):,}")

    print("[3/7] Classifying all candidates from causal history only...")
    labels = build_causal_context_labels(candidates, ledger)
    counts = context_counts(labels)
    for _, r in counts[counts["segment"] == "ALL"].iterrows():
        print(
            f"      {r['definition']:43s} "
            f"N={int(r['reaccel_context_candidates']):4d} "
            f"({100*safe_float(r['reaccel_context_share']):5.1f}%)"
        )

    print("[4/7] Auditing against confident common-50 FIRST/REACCEL labels...")
    structural = build_structural_roles(associations)
    agreement = structural_agreement(labels, structural)
    for _, r in agreement.sort_values(
        "balanced_accuracy", ascending=False
    ).iterrows():
        print(
            f"      {r['definition']:43s} "
            f"recall={100*safe_float(r['reaccel_recall']):5.1f}% "
            f"FIRST-spec={100*safe_float(r['first_specificity']):5.1f}% "
            f"bal-acc={100*safe_float(r['balanced_accuracy']):5.1f}%"
        )

    print("[5/7] Running PRE-COMPUTE common-support feasibility gate...")
    mechanics3 = prepare_mechanics_for_feasibility(mechanics)
    fs_parts = []
    cell_parts = []
    for definition in DEFINITIONS:
        fs, cells = feasibility_for_definition(
            mechanics3, labels, definition
        )
        fs_parts.append(fs)
        cell_parts.append(cells)
    feasibility = pd.concat(fs_parts, ignore_index=True)
    cell_detail = pd.concat(cell_parts, ignore_index=True)
    feasibility_agg = aggregate_feasibility(feasibility)

    for _, r in feasibility_agg[
        feasibility_agg["segment"] == "ALL"
    ].iterrows():
        print(
            f"      {r['definition']:43s} "
            f"core={int(r['core_features_feasible'])}/"
            f"{int(r['core_features'])} "
            f"minCS={int(r['minimum_common_support_n']):4d} "
            f"medianCS={safe_float(r['median_common_support_n']):6.0f} "
            f"FEASIBLE={bool(r['segment_feasible'])}"
        )

    timing = timing_profile(labels)
    ranked = ranked_review_table(
        agreement, feasibility_agg, counts
    )

    print("[6/7] Running invariants...")
    invariants = build_invariants(
        candidates, outcomes, ledger, labels, mechanics3,
        structural, k2_meta, l1_meta,
    )
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    print("[7/7] Writing atomic outputs...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    labels_out = labels.merge(
        structural,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    labels_out["structural_role"] = labels_out[
        "structural_role"
    ].fillna("UNASSOCIATED")

    atomic_write_parquet(
        labels_out,
        args.output_root / "candidate_context_labels.parquet",
    )
    atomic_write_csv(
        counts,
        args.report_root / "context_definition_counts.csv",
    )
    atomic_write_csv(
        agreement,
        args.report_root / "structural_agreement.csv",
    )
    atomic_write_csv(
        feasibility,
        args.report_root / "feasibility_summary.csv",
    )
    atomic_write_csv(
        cell_detail,
        args.report_root / "feasibility_cell_detail.csv",
    )
    atomic_write_csv(
        feasibility_agg,
        args.report_root / "feasibility_gate.csv",
    )
    atomic_write_csv(
        timing,
        args.report_root / "context_timing_profile.csv",
    )
    atomic_write_csv(
        ranked,
        args.report_root / "definition_review_ranking.csv",
    )
    atomic_write_csv(
        invariants,
        args.report_root / "analysis_invariants.csv",
    )

    candidate_hash = sha256_ids(
        candidates["candidate_id"].astype(str)
    )
    feasible_defs = (
        feasibility_agg[
            (feasibility_agg["segment"] == "ALL")
            & feasibility_agg["segment_feasible"].astype(bool)
        ]["definition"].astype(str).tolist()
    )
    top_review = (
        str(ranked.iloc[0]["definition"])
        if len(ranked) and bool(ranked.iloc[0]["review_eligible"])
        else None
    )

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "built_at_utc": utc_now_iso(),
        "candidate_policy": FROZEN_POLICY,
        "candidate_count": int(len(candidates)),
        "candidate_id_set_sha256": candidate_hash,
        "definitions": list(DEFINITIONS),
        "prior50_lookback_sec": PRIOR50_LOOKBACK_SEC,
        "primary_observation_horizon_sec": PRIMARY_OBSERVATION_HORIZON_SEC,
        "primary_feasibility_common_support_n": PRIMARY_FEASIBILITY_N,
        "min_core_features_pass": MIN_CORE_FEATURES_PASS,
        "feasible_definitions": feasible_defs,
        "top_definition_for_human_review": top_review,
        "definition_selection_used_current_75_100_outcomes": False,
        "raw_depth_replay": False,
        "raw_trade_replay": False,
        "failed_invariants": int(len(failed)),
        "scientific_state": (
            "CAUSAL_REACCEL_CONTEXT_AUDITED_NOT_YET_FROZEN"
        ),
    }
    run_config = {
        **summary,
        "phase7k2_root": str(args.phase7k2_root),
        "phase7k4_root": str(args.phase7k4_root),
        "phase7l1_root": str(args.phase7l1_root),
        "causal_history_rule": (
            "prior candidate 50-hit may classify current candidate only if "
            "hit_second <= current origin_second"
        ),
        "structural_audit_rule": (
            "Phase7K4 confident FIRST/REACCEL only; UNASSOCIATED is not negative"
        ),
        "feasibility_rule": (
            "3s mechanics; global momentum/aggression/book terciles; each "
            "momentum x aggression cell requires >=10 high and >=10 low; "
            "common_support=2*sum(min(high,low)); primary >=250 for >=3/4 "
            "core features"
        ),
    }
    atomic_write_json(
        args.output_root / "metadata" / "latest_summary.json",
        summary,
    )
    atomic_write_json(
        args.output_root / "metadata" / "run_config.json",
        run_config,
    )
    atomic_write_text(
        args.report_root / "findings.md",
        findings_text(
            status, counts, agreement, feasibility_agg, ranked
        ),
    )

    print("")
    print(f"Phase 7L.3 status: {status}")
    print(f"Feasible definitions: {feasible_defs}")
    print(
        "Top definition for HUMAN REVIEW only: "
        f"{top_review or 'NONE'}"
    )
    print(
        "No definition was selected using current-candidate 75/100 outcomes."
    )
    print(f"Reports: {args.report_root}")

    if len(failed):
        print("")
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
