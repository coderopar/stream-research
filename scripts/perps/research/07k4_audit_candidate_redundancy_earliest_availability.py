#!/usr/bin/env python3
"""
Phase 7K.4 — Candidate redundancy + earliest causal availability audit
======================================================================

Purpose
-------
Resolve the last two cheap ontology questions before expensive participant /
raw-depth replay:

1. REDUNDANCY
   Does the zero-cooldown 7K.2 branch create repeated causal candidates for
   what hindsight identifies as one common 50-bp expansion impulse?

2. EARLIEST AVAILABILITY
   For common 50-bp impulses that later reach 75 or 100 bps, when was the
   FIRST legitimate causal candidate available?  Do not use the latest
   one-to-one association used by earlier coverage reports to infer timing.

Frozen inputs
-------------
* Phase 7K.2 candidate universes (0 / 60 / 120 / 180 sec cooldowns)
* Frozen Phase 7I.1 candidate universe as historical benchmark
* Phase 7K.3 common-50 impulse ontology with nested 50 / 75 / 100 severity
  labels

No candidate generation, raw-depth replay, raw-trade replay, ML, or options
analysis occurs in this phase.

Association model
-----------------
Each common impulse has an ONSET WINDOW:

    first impulse in a Phase-7A leg:
        earliest frozen parent anchor -> current impulse 50-bp hit

    re-acceleration:
        previous common impulse 50-bp hit -> current impulse 50-bp hit

A candidate counts as an onset candidate only if its causal origin and firing
time belong to that window.  For re-accelerations:

    previous 50 hit < candidate origin <= [release] <= candidate <= current 50 hit

This gives one clean place to measure duplicate attempts.  Candidates after the
current 50-bp hit are NOT called duplicate onset candidates; they are tracked
separately as continuation / escalation confirmations before 75 or 100 bps.

For each impulse and policy the audit records:
* onset candidate count before 50
* earliest and latest onset candidate
* extra onset candidates beyond the first
* first post-50 candidate before 75
* first post-75 candidate before 100
* earliest ANY valid candidate before 50 / 75 / 100
* lead time from first valid candidate to each target

This directly separates:
    detected by 50
from:
    missed by 50 but became detectable between 50 and 75
from:
    missed by 75 but became detectable between 75 and 100

Primary policy comparison
-------------------------
TEST_COOLDOWN_0S versus TEST_COOLDOWN_60S.
120s, 180s and OLD_7I1_FROZEN remain context benchmarks.

Outputs
-------
data/derived/perps/research/candidate_redundancy_earliest_availability_audit_v1/
    impulse_candidate_availability.parquet
    candidate_onset_associations.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_candidate_redundancy_earliest_availability_audit/
    candidate_redundancy_summary.csv
    earliest_availability_summary.csv
    earliest_availability_by_ordinal.csv
    detection_stage_summary.csv
    zero_vs_sixty_incremental_coverage.csv
    candidate_spacing_buckets.csv
    candidate_day_profile.csv
    candidate_storm_windows.csv
    zero_vs_sixty_geometry_overlap.csv
    invariants.csv
    findings.md

Interpretation discipline
-------------------------
* This is a hindsight association audit of already-frozen causal candidates.
* It does not alter candidate generation.
* A candidate can be useful as an escalation confirmation after 50 even if it
  was not early enough to detect the impulse onset by 50.
* Redundancy is judged primarily inside the onset window, not by arbitrary
  global timestamp proximity.
* No winning cooldown is automatically frozen by the script.
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
from typing import Any, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "candidate_redundancy_earliest_availability_audit_v1"

FROZEN_PHASE7K2_VERSION = "1.0.0"
FROZEN_PHASE7K3_VERSION = "1.0.0"

PRIMARY_POLICY = "TEST_COOLDOWN_0S"
CHALLENGER_POLICY = "TEST_COOLDOWN_60S"
K2_POLICIES = (
    "TEST_COOLDOWN_0S",
    "TEST_COOLDOWN_60S",
    "TEST_COOLDOWN_120S",
    "BENCHMARK_COOLDOWN_180S",
)
OLD_POLICY = "OLD_7I1_FROZEN"
ALL_POLICIES = (*K2_POLICIES, OLD_POLICY)

TARGETS_BPS = (50, 75, 100)
HORIZON_MIN = 15
MAX_ASSOCIATION_LOOKBACK_SEC = 900

SPACING_BUCKETS = (
    (0, 30, "LT_30S"),
    (30, 60, "30_59S"),
    (60, 120, "60_119S"),
    (120, 180, "120_179S"),
    (180, 300, "180_299S"),
    (300, 600, "300_599S"),
    (600, math.inf, "GE_600S"),
)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_PHASE7K2_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "causal_rearm_timing_audit_v1"
)
DEFAULT_PHASE7K3_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "expansion_impulse_severity_threshold_audit_v1"
)
DEFAULT_PHASE7I1_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "causal_large_move_candidate_catalog_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_candidate_redundancy_earliest_availability_audit"
)


def require_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "Phase 7K.4 requires numpy and pandas. "
            "Activate the project .venv first."
        ) from exc
    return np, pd, None, None


def require_parquet_stack():
    np, pd, _, _ = require_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Phase 7K.4 Parquet I/O requires pyarrow. "
            "Activate the project .venv first."
        ) from exc
    return np, pd, pa, pq


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_time(sec: Any) -> str | None:
    try:
        x = float(sec)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return datetime.fromtimestamp(
        int(x), tz=timezone.utc
    ).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7k4-", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
    )


def atomic_write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7k4-", suffix=".csv", dir=str(path.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_parquet(df, path: Path) -> None:
    _, _, pa, pq = require_parquet_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7k4-", suffix=".parquet", dir=str(path.parent)
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


def load_metadata(root: Path, expected_version: str, label: str) -> dict[str, Any]:
    path = root / "metadata" / "latest_summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing {label} metadata: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if str(meta.get("builder_version")) != expected_version:
        raise RuntimeError(
            f"Phase 7K.4 requires {label} builder {expected_version}; "
            f"found {meta.get('builder_version')!r}"
        )
    if str(meta.get("status", "PASS")) != "PASS":
        raise RuntimeError(f"Required {label} did not finish PASS: {meta.get('status')!r}")
    return meta


def load_inputs(k2_root: Path, k3_root: Path, old_root: Path):
    _, pd, _, pq = require_parquet_stack()

    k2_meta = load_metadata(k2_root, FROZEN_PHASE7K2_VERSION, "Phase-7K.2")
    k3_meta = load_metadata(k3_root, FROZEN_PHASE7K3_VERSION, "Phase-7K.3")

    k2_path = k2_root / "candidates.parquet"
    impulses_path = k3_root / "common_impulses.parquet"
    parents_path = k3_root / "parent_detections.parquet"
    old_path = old_root / "candidates.parquet"

    missing = [
        str(p) for p in (k2_path, impulses_path, parents_path, old_path)
        if not p.exists()
    ]
    if missing:
        raise RuntimeError("Missing required inputs:\n  " + "\n  ".join(missing))

    k2 = pq.read_table(k2_path).to_pandas()
    impulses = pq.read_table(impulses_path).to_pandas()
    parents = pq.read_table(parents_path).to_pandas()
    old = pq.read_table(old_path).to_pandas()

    required_k2 = {
        "candidate_id", "policy_id", "direction", "origin_second",
        "candidate_second", "release_second",
    }
    missing_k2 = required_k2 - set(k2.columns)
    if missing_k2:
        raise RuntimeError(f"Phase-7K.2 candidates missing: {sorted(missing_k2)}")

    required_imp = {
        "impulse_id", "direction", "representative_anchor_second",
        "hit_50_second", "hit_75_second", "hit_100_second",
        "reach_50_15m", "reach_75_15m", "reach_100_15m",
        "phase7a_event_id", "impulse_ordinal_in_physical_leg",
        "previous_impulse_50_hit_second", "is_reacceleration",
    }
    missing_imp = required_imp - set(impulses.columns)
    if missing_imp:
        raise RuntimeError(f"Phase-7K.3 common impulses missing: {sorted(missing_imp)}")

    required_parent = {"impulse_id", "anchor_second"}
    missing_parent = required_parent - set(parents.columns)
    if missing_parent:
        raise RuntimeError(f"Phase-7K.3 parent detections missing: {sorted(missing_parent)}")

    required_old = {
        "candidate_id", "direction", "origin_second", "candidate_second",
    }
    missing_old = required_old - set(old.columns)
    if missing_old:
        raise RuntimeError(f"Frozen 7I.1 candidates missing: {sorted(missing_old)}")

    old = old.copy()
    old["policy_id"] = OLD_POLICY
    if "release_second" not in old.columns:
        old["release_second"] = math.nan

    candidates = pd.concat([k2, old], ignore_index=True, sort=False)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    return k2_meta, k3_meta, candidates, impulses, parents


def add_window_starts(impulses, parents):
    _, pd, _, _ = require_stack()
    min_anchor = (
        parents.groupby("impulse_id")["anchor_second"]
        .min()
        .to_dict()
    )

    out = impulses.copy()
    out["earliest_parent_anchor_second"] = out["impulse_id"].astype(str).map(
        lambda x: min_anchor.get(x, math.nan)
    )

    starts = []
    kinds = []
    for r in out.to_dict("records"):
        prev = safe_float(r.get("previous_impulse_50_hit_second"))
        ordinal = safe_float(r.get("impulse_ordinal_in_physical_leg"))
        if math.isfinite(ordinal) and int(ordinal) > 1 and math.isfinite(prev):
            starts.append(int(prev))
            kinds.append("AFTER_PREVIOUS_COMMON_50_HIT")
        else:
            a = safe_float(r.get("earliest_parent_anchor_second"))
            if not math.isfinite(a):
                a = safe_float(r.get("representative_anchor_second"))
            starts.append(int(a))
            kinds.append("EARLIEST_COMMON_PARENT_ANCHOR")

    out["onset_window_start_second"] = starts
    out["onset_window_start_time"] = out["onset_window_start_second"].map(iso_time)
    out["onset_window_start_kind"] = kinds
    return out


def candidate_index(candidates):
    _, pd, _, _ = require_stack()
    by_policy_dir: dict[tuple[str, int], tuple[list[int], Any]] = {}

    c = candidates.copy()
    for col in ("candidate_second", "origin_second", "release_second"):
        if col not in c.columns:
            c[col] = math.nan
        c[col] = pd.to_numeric(c[col], errors="coerce")

    c = c[
        c["candidate_second"].notna()
        & c["origin_second"].notna()
        & c["direction"].notna()
    ].copy()

    for (policy, direction), g in c.groupby(
        [c["policy_id"].astype(str), pd.to_numeric(c["direction"], errors="coerce")],
        sort=False,
    ):
        g = g.sort_values("candidate_second", kind="mergesort").reset_index(drop=True)
        secs = g["candidate_second"].astype(int).tolist()
        by_policy_dir[(str(policy), int(direction))] = (secs, g)

    return by_policy_dir


def valid_candidates_in_interval(
    index,
    policy: str,
    direction: int,
    start_sec: int,
    end_sec: int,
    *,
    strict_origin_after_start: bool,
):
    """
    Return candidates whose causal origin belongs to [start,end] (or (start,end]
    for re-acceleration windows) and whose release/candidate ordering is valid.
    """
    _, pd, _, _ = require_stack()
    pack = index.get((str(policy), int(direction)))
    if pack is None or end_sec < start_sec:
        return pd.DataFrame()

    secs, g = pack
    lo = bisect_left(secs, int(start_sec) - MAX_ASSOCIATION_LOOKBACK_SEC)
    hi = bisect_right(secs, int(end_sec))
    if lo >= hi:
        return g.iloc[0:0].copy()

    x = g.iloc[lo:hi].copy()
    origin = pd.to_numeric(x["origin_second"], errors="coerce")
    cand = pd.to_numeric(x["candidate_second"], errors="coerce")
    release = pd.to_numeric(x["release_second"], errors="coerce")
    have_release = release.notna()

    if strict_origin_after_start:
        start_ok = origin > int(start_sec)
    else:
        start_ok = origin >= int(start_sec)

    ordering = (
        start_ok
        & (origin <= cand)
        & (cand <= int(end_sec))
        & (~have_release | ((origin <= release) & (release <= cand)))
    )
    return x.loc[ordering].sort_values("candidate_second", kind="mergesort").reset_index(drop=True)


def build_availability(impulses, candidates):
    _, pd, _, _ = require_stack()
    idx = candidate_index(candidates)
    summary_rows = []
    onset_detail_rows = []

    for n, imp in enumerate(impulses.to_dict("records"), start=1):
        if n % 500 == 0:
            print(f"      associated impulses={n:,}/{len(impulses):,}")

        direction = int(imp["direction"])
        start = int(imp["onset_window_start_second"])
        hit50 = int(imp["hit_50_second"])
        ordinal_raw = safe_float(imp.get("impulse_ordinal_in_physical_leg"))
        ordinal = int(ordinal_raw) if math.isfinite(ordinal_raw) else None
        is_reacc = bool(ordinal is not None and ordinal > 1)
        strict = is_reacc

        for policy in ALL_POLICIES:
            onset = valid_candidates_in_interval(
                idx, policy, direction, start, hit50,
                strict_origin_after_start=strict,
            )

            for j, r in onset.iterrows():
                onset_detail_rows.append({
                    "policy_id": policy,
                    "impulse_id": str(imp["impulse_id"]),
                    "phase7a_event_id": imp.get("phase7a_event_id"),
                    "impulse_ordinal_in_physical_leg": (
                        ordinal if ordinal is not None else math.nan
                    ),
                    "is_reacceleration": is_reacc,
                    "direction": direction,
                    "onset_window_start_second": start,
                    "hit_50_second": hit50,
                    "candidate_rank_in_onset_window": int(j + 1),
                    "candidate_id": str(r["candidate_id"]),
                    "candidate_origin_second": int(r["origin_second"]),
                    "candidate_release_second": (
                        int(r["release_second"])
                        if math.isfinite(safe_float(r.get("release_second")))
                        else math.nan
                    ),
                    "candidate_second": int(r["candidate_second"]),
                    "candidate_lead_to_50_sec": int(hit50 - int(r["candidate_second"])),
                })

            row = {
                "policy_id": policy,
                "impulse_id": str(imp["impulse_id"]),
                "phase7a_event_id": imp.get("phase7a_event_id"),
                "impulse_ordinal_in_physical_leg": (
                    ordinal if ordinal is not None else math.nan
                ),
                "is_reacceleration": is_reacc,
                "direction": direction,
                "direction_label": imp.get("direction_label"),
                "onset_window_start_second": start,
                "onset_window_start_time": iso_time(start),
                "hit_50_second": hit50,
                "hit_50_time": iso_time(hit50),
                "reach_50_15m": bool(imp.get("reach_50_15m", False)),
                "reach_75_15m": bool(imp.get("reach_75_15m", False)),
                "reach_100_15m": bool(imp.get("reach_100_15m", False)),
                "onset_candidate_count": int(len(onset)),
                "extra_onset_candidates": max(0, int(len(onset)) - 1),
                "onset_redundant": bool(len(onset) >= 2),
            }

            if len(onset):
                first = onset.iloc[0]
                last = onset.iloc[-1]
                row.update({
                    "first_onset_candidate_id": str(first["candidate_id"]),
                    "first_onset_candidate_second": int(first["candidate_second"]),
                    "first_onset_candidate_time": iso_time(int(first["candidate_second"])),
                    "first_onset_lead_to_50_sec": int(hit50 - int(first["candidate_second"])),
                    "latest_onset_candidate_id": str(last["candidate_id"]),
                    "latest_onset_candidate_second": int(last["candidate_second"]),
                    "latest_onset_lead_to_50_sec": int(hit50 - int(last["candidate_second"])),
                })
                if len(onset) >= 2:
                    row["first_to_second_onset_gap_sec"] = int(
                        int(onset.iloc[1]["candidate_second"])
                        - int(onset.iloc[0]["candidate_second"])
                    )
                else:
                    row["first_to_second_onset_gap_sec"] = math.nan
            else:
                row.update({
                    "first_onset_candidate_id": None,
                    "first_onset_candidate_second": math.nan,
                    "first_onset_candidate_time": None,
                    "first_onset_lead_to_50_sec": math.nan,
                    "latest_onset_candidate_id": None,
                    "latest_onset_candidate_second": math.nan,
                    "latest_onset_lead_to_50_sec": math.nan,
                    "first_to_second_onset_gap_sec": math.nan,
                })

            # Target-aware availability.  We deliberately do NOT use one-to-one
            # matching here: the question is when the first valid causal
            # candidate existed, not which candidate a greedy matcher chose.
            for target in TARGETS_BPS:
                reach = bool(imp.get(f"reach_{target}_15m", False))
                hit_raw = safe_float(imp.get(f"hit_{target}_second"))
                hit = int(hit_raw) if reach and math.isfinite(hit_raw) else None

                row[f"eligible_{target}"] = bool(hit is not None)
                if hit is None:
                    row[f"candidate_count_by_{target}"] = 0
                    row[f"first_candidate_by_{target}_second"] = math.nan
                    row[f"first_candidate_lead_to_{target}_sec"] = math.nan
                    row[f"detection_stage_by_{target}"] = "NOT_ELIGIBLE"
                    continue

                all_by_target = valid_candidates_in_interval(
                    idx, policy, direction, start, hit,
                    strict_origin_after_start=strict,
                )
                row[f"candidate_count_by_{target}"] = int(len(all_by_target))

                if len(all_by_target):
                    first_target = all_by_target.iloc[0]
                    first_sec = int(first_target["candidate_second"])
                    row[f"first_candidate_by_{target}_second"] = first_sec
                    row[f"first_candidate_lead_to_{target}_sec"] = int(hit - first_sec)

                    if first_sec <= hit50:
                        stage = "BY_50_ONSET"
                    elif target >= 75 and first_sec <= int(imp["hit_75_second"]):
                        stage = "BETWEEN_50_AND_75"
                    elif target >= 100:
                        stage = "BETWEEN_75_AND_100"
                    else:
                        stage = "AFTER_50_BEFORE_TARGET"
                    row[f"detection_stage_by_{target}"] = stage
                else:
                    row[f"first_candidate_by_{target}_second"] = math.nan
                    row[f"first_candidate_lead_to_{target}_sec"] = math.nan
                    row[f"detection_stage_by_{target}"] = "UNDETECTED"

            summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(onset_detail_rows)


def redundancy_summary(availability):
    _, pd, _, _ = require_stack()
    rows = []
    for (policy, reacc), g in availability.groupby(
        ["policy_id", "is_reacceleration"], sort=False
    ):
        covered = g[g["onset_candidate_count"] > 0]
        redundant = g[g["onset_candidate_count"] >= 2]
        rows.append({
            "policy_id": policy,
            "segment": "REACCEL" if bool(reacc) else "FIRST_OR_UNMAPPED",
            "impulses": int(len(g)),
            "onset_covered_impulses": int(len(covered)),
            "onset_coverage_rate": float(len(covered) / len(g)) if len(g) else math.nan,
            "redundant_impulses_2plus_candidates": int(len(redundant)),
            "redundant_share_of_covered": (
                float(len(redundant) / len(covered)) if len(covered) else math.nan
            ),
            "total_onset_candidates": int(g["onset_candidate_count"].sum()),
            "extra_onset_candidates": int(g["extra_onset_candidates"].sum()),
            "extra_candidates_per_covered_impulse": (
                float(g["extra_onset_candidates"].sum() / len(covered))
                if len(covered) else math.nan
            ),
            "median_candidates_per_covered_impulse": (
                float(covered["onset_candidate_count"].median())
                if len(covered) else math.nan
            ),
            "max_candidates_single_impulse": int(g["onset_candidate_count"].max()) if len(g) else 0,
        })
    return pd.DataFrame(rows)


def earliest_summary(availability):
    _, pd, _, _ = require_stack()
    rows = []
    for policy in ALL_POLICIES:
        p = availability[availability["policy_id"] == policy]
        for target in TARGETS_BPS:
            eligible = p[p[f"eligible_{target}"].fillna(False).astype(bool)].copy()
            for segment, mask in (
                ("ALL", pd.Series(True, index=eligible.index)),
                ("REACCEL", eligible["is_reacceleration"].fillna(False).astype(bool)),
            ):
                g = eligible.loc[mask]
                detected = g[pd.to_numeric(g[f"first_candidate_by_{target}_second"], errors="coerce").notna()]
                lead = pd.to_numeric(
                    detected[f"first_candidate_lead_to_{target}_sec"], errors="coerce"
                )
                rows.append({
                    "policy_id": policy,
                    "target_bps": target,
                    "horizon_min": HORIZON_MIN,
                    "segment": segment,
                    "eligible_impulses": int(len(g)),
                    "detected_impulses": int(len(detected)),
                    "coverage_rate": float(len(detected) / len(g)) if len(g) else math.nan,
                    "median_first_candidate_lead_sec": float(lead.median()) if len(lead) else math.nan,
                    "p25_first_candidate_lead_sec": float(lead.quantile(0.25)) if len(lead) else math.nan,
                    "p75_first_candidate_lead_sec": float(lead.quantile(0.75)) if len(lead) else math.nan,
                    "share_first_candidate_ge_60s_before_target": float((lead >= 60).mean()) if len(lead) else math.nan,
                    "share_first_candidate_ge_120s_before_target": float((lead >= 120).mean()) if len(lead) else math.nan,
                })
    return pd.DataFrame(rows)


def earliest_by_ordinal(availability):
    _, pd, _, _ = require_stack()
    rows = []
    x = availability.copy()
    x["ordinal"] = pd.to_numeric(
        x["impulse_ordinal_in_physical_leg"], errors="coerce"
    )
    for (policy, ordinal), g0 in x.groupby(["policy_id", "ordinal"], dropna=False):
        for target in TARGETS_BPS:
            g = g0[g0[f"eligible_{target}"].fillna(False).astype(bool)]
            detected = g[pd.to_numeric(g[f"first_candidate_by_{target}_second"], errors="coerce").notna()]
            lead = pd.to_numeric(detected[f"first_candidate_lead_to_{target}_sec"], errors="coerce")
            rows.append({
                "policy_id": policy,
                "impulse_ordinal_in_physical_leg": ordinal,
                "target_bps": target,
                "eligible_impulses": int(len(g)),
                "detected_impulses": int(len(detected)),
                "coverage_rate": float(len(detected) / len(g)) if len(g) else math.nan,
                "median_first_candidate_lead_sec": float(lead.median()) if len(lead) else math.nan,
            })
    return pd.DataFrame(rows)


def detection_stage_summary(availability):
    _, pd, _, _ = require_stack()
    rows = []
    for policy in ALL_POLICIES:
        p = availability[availability["policy_id"] == policy]
        for target in TARGETS_BPS:
            eligible = p[p[f"eligible_{target}"].fillna(False).astype(bool)].copy()
            for segment, mask in (
                ("ALL", pd.Series(True, index=eligible.index)),
                ("REACCEL", eligible["is_reacceleration"].fillna(False).astype(bool)),
            ):
                g = eligible.loc[mask]
                counts = g[f"detection_stage_by_{target}"].value_counts(dropna=False).to_dict()
                total = len(g)
                for stage in (
                    "BY_50_ONSET",
                    "BETWEEN_50_AND_75",
                    "BETWEEN_75_AND_100",
                    "AFTER_50_BEFORE_TARGET",
                    "UNDETECTED",
                ):
                    n = int(counts.get(stage, 0))
                    rows.append({
                        "policy_id": policy,
                        "target_bps": target,
                        "segment": segment,
                        "detection_stage": stage,
                        "impulses": n,
                        "share": float(n / total) if total else math.nan,
                    })
    return pd.DataFrame(rows)


def zero_vs_sixty_incremental(availability):
    _, pd, _, _ = require_stack()
    a = availability[availability["policy_id"] == PRIMARY_POLICY].copy()
    b = availability[availability["policy_id"] == CHALLENGER_POLICY].copy()
    rows = []

    for target in TARGETS_BPS:
        cols = [
            "impulse_id", "is_reacceleration", f"eligible_{target}",
            f"first_candidate_by_{target}_second", f"first_candidate_lead_to_{target}_sec",
        ]
        aa = a[cols].rename(columns={
            f"eligible_{target}": "eligible_0",
            f"first_candidate_by_{target}_second": "first_0",
            f"first_candidate_lead_to_{target}_sec": "lead_0",
        })
        bb = b[cols].rename(columns={
            "is_reacceleration": "is_reacceleration_60",
            f"eligible_{target}": "eligible_60",
            f"first_candidate_by_{target}_second": "first_60",
            f"first_candidate_lead_to_{target}_sec": "lead_60",
        })
        m = aa.merge(bb, on="impulse_id", how="inner")
        m = m[m["eligible_0"].fillna(False).astype(bool) & m["eligible_60"].fillna(False).astype(bool)]
        d0 = pd.to_numeric(m["first_0"], errors="coerce").notna()
        d60 = pd.to_numeric(m["first_60"], errors="coerce").notna()

        for segment, mask in (
            ("ALL", pd.Series(True, index=m.index)),
            ("REACCEL", m["is_reacceleration"].fillna(False).astype(bool)),
        ):
            x = m.loc[mask]
            x0 = d0.loc[x.index]
            x60 = d60.loc[x.index]
            both = x0 & x60
            lead_adv = (
                pd.to_numeric(x.loc[both, "lead_0"], errors="coerce")
                - pd.to_numeric(x.loc[both, "lead_60"], errors="coerce")
            )
            rows.append({
                "target_bps": target,
                "segment": segment,
                "eligible_impulses": int(len(x)),
                "covered_0s": int(x0.sum()),
                "covered_60s": int(x60.sum()),
                "zero_only": int((x0 & ~x60).sum()),
                "sixty_only": int((~x0 & x60).sum()),
                "both": int(both.sum()),
                "neither": int((~x0 & ~x60).sum()),
                "net_coverage_0s_minus_60s": int(x0.sum() - x60.sum()),
                "median_extra_lead_sec_0s_when_both": float(lead_adv.median()) if len(lead_adv) else math.nan,
                "share_0s_earlier_when_both": float((lead_adv > 0).mean()) if len(lead_adv) else math.nan,
            })

    return pd.DataFrame(rows)


def spacing_bucket_name(gap: float) -> str:
    for lo, hi, name in SPACING_BUCKETS:
        if gap >= lo and gap < hi:
            return name
    return "UNKNOWN"


def spacing_buckets(candidates):
    _, pd, _, _ = require_stack()
    rows = []
    for (policy, direction), g in candidates.groupby(
        [candidates["policy_id"].astype(str), pd.to_numeric(candidates["direction"], errors="coerce")],
        sort=False,
    ):
        secs = sorted(pd.to_numeric(g["candidate_second"], errors="coerce").dropna().astype(int))
        gaps = [b - a for a, b in zip(secs, secs[1:])]
        counts: dict[str, int] = {}
        for gap in gaps:
            counts[spacing_bucket_name(float(gap))] = counts.get(spacing_bucket_name(float(gap)), 0) + 1
        for _, _, name in SPACING_BUCKETS:
            n = int(counts.get(name, 0))
            rows.append({
                "policy_id": str(policy),
                "direction": int(direction),
                "direction_label": "UP" if int(direction) > 0 else "DOWN",
                "gap_bucket": name,
                "gaps": n,
                "share": float(n / len(gaps)) if gaps else math.nan,
            })
    return pd.DataFrame(rows)


def day_and_storm_profiles(candidates):
    _, pd, _, _ = require_stack()
    c = candidates.copy()
    c["candidate_second"] = pd.to_numeric(c["candidate_second"], errors="coerce")
    c = c[c["candidate_second"].notna()].copy()
    c["candidate_dt"] = pd.to_datetime(c["candidate_second"].astype(int), unit="s", utc=True)
    c["candidate_date"] = c["candidate_dt"].dt.strftime("%Y-%m-%d")
    c["hour_start"] = c["candidate_dt"].dt.floor("h")

    day = (
        c.groupby(["policy_id", "candidate_date"])
        .size().rename("candidates").reset_index()
    )
    day["rank_within_policy"] = day.groupby("policy_id")["candidates"].rank(
        method="first", ascending=False
    ).astype(int)

    storm = (
        c.groupby(["policy_id", "hour_start"])
        .size().rename("candidates_in_hour").reset_index()
    )
    storm["rank_within_policy"] = storm.groupby("policy_id")["candidates_in_hour"].rank(
        method="first", ascending=False
    ).astype(int)
    storm = storm[storm["rank_within_policy"] <= 20].sort_values(
        ["policy_id", "rank_within_policy"], kind="mergesort"
    )
    return day, storm


def geometry_overlap(candidates):
    _, pd, _, _ = require_stack()
    a = candidates[candidates["policy_id"] == PRIMARY_POLICY].copy()
    b = candidates[candidates["policy_id"] == CHALLENGER_POLICY].copy()

    def keys(g):
        out = set()
        for r in g.to_dict("records"):
            rel = safe_float(r.get("release_second"))
            rel_key = int(rel) if math.isfinite(rel) else None
            out.add((
                int(r["direction"]),
                int(r["origin_second"]),
                rel_key,
                int(r["candidate_second"]),
            ))
        return out

    ka = keys(a)
    kb = keys(b)
    rows = [{
        "comparison": "0S_VS_60S_EXACT_CAUSAL_GEOMETRY",
        "zero_candidates": int(len(ka)),
        "sixty_candidates": int(len(kb)),
        "shared_geometry": int(len(ka & kb)),
        "zero_only_geometry": int(len(ka - kb)),
        "sixty_only_geometry": int(len(kb - ka)),
        "zero_share_shared": float(len(ka & kb) / len(ka)) if ka else math.nan,
        "sixty_share_shared": float(len(ka & kb) / len(kb)) if kb else math.nan,
    }]
    return pd.DataFrame(rows)


def build_invariants(availability, onset_detail, candidates, impulses):
    _, pd, _, _ = require_stack()
    rows = []

    def add(name, ok, observed, expected):
        rows.append({
            "invariant": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    expected_rows = len(impulses) * len(ALL_POLICIES)
    add(
        "one_availability_row_per_impulse_policy",
        len(availability) == expected_rows,
        len(availability),
        expected_rows,
    )
    add(
        "availability_impulse_policy_unique",
        not availability.duplicated(["impulse_id", "policy_id"]).any(),
        int(availability.duplicated(["impulse_id", "policy_id"]).sum()),
        0,
    )

    bad_window = (
        pd.to_numeric(availability["onset_window_start_second"], errors="coerce")
        >= pd.to_numeric(availability["hit_50_second"], errors="coerce")
    )
    add(
        "onset_windows_positive_duration",
        not bool(bad_window.any()),
        int(bad_window.sum()),
        0,
    )

    if len(onset_detail):
        origin = pd.to_numeric(onset_detail["candidate_origin_second"], errors="coerce")
        cand = pd.to_numeric(onset_detail["candidate_second"], errors="coerce")
        start = pd.to_numeric(onset_detail["onset_window_start_second"], errors="coerce")
        hit50 = pd.to_numeric(onset_detail["hit_50_second"], errors="coerce")
        reacc = onset_detail["is_reacceleration"].fillna(False).astype(bool)
        start_ok = (~reacc & (origin >= start)) | (reacc & (origin > start))
        ok = start_ok & (origin <= cand) & (cand <= hit50)
        add(
            "all_onset_associations_belong_to_onset_window",
            bool(ok.all()),
            int((~ok).sum()),
            0,
        )

    # Target coverage must be nested for the same policy/impulse because the
    # same candidate search window only expands as the severity target rises.
    nesting_violations = 0
    for _, r in availability.iterrows():
        c50 = math.isfinite(safe_float(r.get("first_candidate_by_50_second")))
        c75 = math.isfinite(safe_float(r.get("first_candidate_by_75_second")))
        c100 = math.isfinite(safe_float(r.get("first_candidate_by_100_second")))
        e75 = bool(r.get("eligible_75", False))
        e100 = bool(r.get("eligible_100", False))
        if e75 and c50 and not c75:
            nesting_violations += 1
        if e100 and c75 and not c100:
            nesting_violations += 1
    add(
        "candidate_availability_nested_as_target_window_expands",
        nesting_violations == 0,
        nesting_violations,
        0,
    )

    # 0s should contain at least as many source candidates as 60s; this is a
    # structural expectation of the frozen 7K.2 branches, not a performance gate.
    counts = candidates.groupby("policy_id").size().to_dict()
    n0 = int(counts.get(PRIMARY_POLICY, 0))
    n60 = int(counts.get(CHALLENGER_POLICY, 0))
    add("zero_source_candidate_count_ge_sixty", n0 >= n60, n0, f">={n60}")

    return pd.DataFrame(rows)


def findings_text(status, redundancy, earliest, incremental, geometry, storm):
    def get_red(policy, segment="REACCEL"):
        x = redundancy[(redundancy["policy_id"] == policy) & (redundancy["segment"] == segment)]
        return x.iloc[0] if len(x) else None

    lines = [
        "# Phase 7K.4 — Candidate redundancy + earliest causal availability audit",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{status}**",
        f"- Primary comparison: **{PRIMARY_POLICY} vs {CHALLENGER_POLICY}**",
        "- Impulse ontology: **frozen common-50 Phase 7K.3**",
        "",
        "## Onset-window redundancy",
        "",
    ]

    for policy in (PRIMARY_POLICY, CHALLENGER_POLICY, "BENCHMARK_COOLDOWN_180S", OLD_POLICY):
        r = get_red(policy)
        if r is not None:
            lines.append(
                f"- {policy}: reaccel onset coverage={100*safe_float(r['onset_coverage_rate']):.1f}%; "
                f"2+ onset candidates among covered={100*safe_float(r['redundant_share_of_covered']):.1f}%; "
                f"extra onset candidates={int(r['extra_onset_candidates']):,}."
            )

    lines += ["", "## Earliest availability for re-accelerations", ""]
    for target in TARGETS_BPS:
        for policy in (PRIMARY_POLICY, CHALLENGER_POLICY):
            x = earliest[
                (earliest["policy_id"] == policy)
                & (earliest["target_bps"] == target)
                & (earliest["segment"] == "REACCEL")
            ]
            if len(x):
                r = x.iloc[0]
                lines.append(
                    f"- {policy} @ {target}bp: coverage={100*safe_float(r['coverage_rate']):.1f}%; "
                    f"median first-candidate lead={safe_float(r['median_first_candidate_lead_sec']):.0f}s."
                )

    lines += ["", "## 0s versus 60s incremental value", ""]
    for target in TARGETS_BPS:
        x = incremental[(incremental["target_bps"] == target) & (incremental["segment"] == "REACCEL")]
        if len(x):
            r = x.iloc[0]
            lines.append(
                f"- {target}bp reaccels: 0s-only={int(r['zero_only'])}; "
                f"60s-only={int(r['sixty_only'])}; net={int(r['net_coverage_0s_minus_60s']):+d}; "
                f"median extra lead when both={safe_float(r['median_extra_lead_sec_0s_when_both']):+.0f}s."
            )

    if len(geometry):
        r = geometry.iloc[0]
        lines += [
            "",
            "## Exact candidate geometry overlap",
            "",
            f"- Shared 0s/60s causal geometries: {int(r['shared_geometry']):,}.",
            f"- 0s-only geometries: {int(r['zero_only_geometry']):,}.",
            f"- 60s-only geometries: {int(r['sixty_only_geometry']):,}.",
        ]

    if len(storm):
        s0 = storm[(storm["policy_id"] == PRIMARY_POLICY)].sort_values("candidates_in_hour", ascending=False)
        s60 = storm[(storm["policy_id"] == CHALLENGER_POLICY)].sort_values("candidates_in_hour", ascending=False)
        if len(s0) and len(s60):
            lines += [
                "",
                "## Peak hourly candidate concentration",
                "",
                f"- 0s maximum: {int(s0.iloc[0]['candidates_in_hour'])} candidates/hour.",
                f"- 60s maximum: {int(s60.iloc[0]['candidates_in_hour'])} candidates/hour.",
            ]

    lines += [
        "",
        "## Decision discipline",
        "",
        "- Prefer 0s only if its incremental 75/100 coverage or earlier availability is meaningful relative to its extra candidate redundancy.",
        "- If most 0s-only candidates are redundant onset attempts and 60s preserves nearly all early 75-bp detection, prefer 60s for the frozen ontology.",
        "- If 0s materially improves first-candidate timing or 100-bp coverage without pathological onset duplication/storms, 0s remains justified.",
        "- Do not infer entry profitability from this audit; it resolves candidate identity and timing only.",
        "- Do not start raw-depth replay until this comparison is reviewed and one candidate ontology is frozen.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    _, pd, _, _ = require_stack()

    impulses = pd.DataFrame([
        {
            "impulse_id": "imp1",
            "direction": 1,
            "direction_label": "UP",
            "phase7a_event_id": "leg1",
            "impulse_ordinal_in_physical_leg": 1,
            "previous_impulse_50_hit_second": math.nan,
            "is_reacceleration": False,
            "representative_anchor_second": 100,
            "earliest_parent_anchor_second": 100,
            "onset_window_start_second": 100,
            "onset_window_start_time": iso_time(100),
            "hit_50_second": 200,
            "hit_75_second": 260,
            "hit_100_second": 320,
            "reach_50_15m": True,
            "reach_75_15m": True,
            "reach_100_15m": True,
        },
        {
            "impulse_id": "imp2",
            "direction": 1,
            "direction_label": "UP",
            "phase7a_event_id": "leg1",
            "impulse_ordinal_in_physical_leg": 2,
            "previous_impulse_50_hit_second": 200,
            "is_reacceleration": True,
            "representative_anchor_second": 300,
            "earliest_parent_anchor_second": 300,
            "onset_window_start_second": 200,
            "onset_window_start_time": iso_time(200),
            "hit_50_second": 400,
            "hit_75_second": 480,
            "hit_100_second": 560,
            "reach_50_15m": True,
            "reach_75_15m": True,
            "reach_100_15m": True,
        },
    ])

    candidates = pd.DataFrame([
        {"candidate_id":"a","policy_id":PRIMARY_POLICY,"direction":1,"origin_second":120,"release_second":140,"candidate_second":170},
        {"candidate_id":"b","policy_id":PRIMARY_POLICY,"direction":1,"origin_second":210,"release_second":225,"candidate_second":250},
        {"candidate_id":"c","policy_id":PRIMARY_POLICY,"direction":1,"origin_second":230,"release_second":245,"candidate_second":270},
        {"candidate_id":"d","policy_id":PRIMARY_POLICY,"direction":1,"origin_second":410,"release_second":430,"candidate_second":450},
        {"candidate_id":"e","policy_id":CHALLENGER_POLICY,"direction":1,"origin_second":210,"release_second":225,"candidate_second":250},
    ])
    # Empty benchmark policies are legal in the synthetic test; add one row well
    # outside the windows so the index paths exist without affecting results.
    for p in ("TEST_COOLDOWN_120S", "BENCHMARK_COOLDOWN_180S", OLD_POLICY):
        candidates.loc[len(candidates)] = {
            "candidate_id": f"x-{p}", "policy_id": p, "direction": -1,
            "origin_second": 1000, "release_second": math.nan,
            "candidate_second": 1010,
        }

    availability, detail = build_availability(impulses, candidates)
    r = availability[
        (availability["policy_id"] == PRIMARY_POLICY)
        & (availability["impulse_id"] == "imp2")
    ].iloc[0]

    assert int(r["onset_candidate_count"]) == 2
    assert int(r["extra_onset_candidates"]) == 1
    assert int(r["first_onset_candidate_second"]) == 250
    assert int(r["first_candidate_by_75_second"]) == 250
    assert str(r["detection_stage_by_75"]) == "BY_50_ONSET"

    # Remove onset candidates: a post-50 candidate should count for 75/100 but
    # must not be called onset coverage.
    late_candidates = candidates[
        ~candidates["candidate_id"].isin(["b", "c", "e"])
    ].copy()
    late_avail, _ = build_availability(impulses, late_candidates)
    late = late_avail[
        (late_avail["policy_id"] == PRIMARY_POLICY)
        & (late_avail["impulse_id"] == "imp2")
    ].iloc[0]
    assert int(late["onset_candidate_count"]) == 0
    assert math.isnan(safe_float(late["first_candidate_by_50_second"]))
    assert int(late["first_candidate_by_75_second"]) == 450
    assert str(late["detection_stage_by_75"]) == "BETWEEN_50_AND_75"

    inv = build_invariants(availability, detail, candidates, impulses)
    assert (inv["status"] == "PASS").all(), inv

    print("SELF-TEST PASS")
    print(
        "Validated common-50 onset windows, duplicate onset counting, earliest "
        "candidate timing, post-50 escalation classification, 0s/60s comparison "
        "plumbing, and nested target availability."
    )


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Phase 7K.4 candidate redundancy + earliest availability audit"
    )
    p.add_argument("--phase7k2-root", type=Path, default=DEFAULT_PHASE7K2_ROOT)
    p.add_argument("--phase7k3-root", type=Path, default=DEFAULT_PHASE7K3_ROOT)
    p.add_argument("--phase7i1-root", type=Path, default=DEFAULT_PHASE7I1_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    _, pd, _, _ = require_stack()

    print("Building Phase 7K.4 candidate redundancy + earliest availability audit...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Required Phase-7K.2   : {FROZEN_PHASE7K2_VERSION}")
    print(f"Required Phase-7K.3   : {FROZEN_PHASE7K3_VERSION}")
    print(f"Primary policy        : {PRIMARY_POLICY}")
    print(f"Challenger policy     : {CHALLENGER_POLICY}")
    print(f"Targets               : {TARGETS_BPS}")
    print(f"Severity horizon      : {HORIZON_MIN}m")
    print("Candidate generation  : NONE")
    print("Raw depth replay      : NONE")
    print("Raw trade replay      : NONE")
    print("Research role         : ontology/timing audit; not alpha")

    print("[1/7] Loading frozen 7K.2 candidates, 7K.3 common impulses and old 7I.1...")
    k2_meta, k3_meta, candidates, impulses, parents = load_inputs(
        args.phase7k2_root, args.phase7k3_root, args.phase7i1_root
    )
    print(
        f"      candidates={len(candidates):,} "
        f"common_impulses={len(impulses):,} parents={len(parents):,}"
    )

    print("[2/7] Freezing common-50 onset windows...")
    impulses = add_window_starts(impulses, parents)
    reacc_n = int(impulses["is_reacceleration"].fillna(False).astype(bool).sum())
    print(f"      common impulses={len(impulses):,} reaccelerations={reacc_n:,}")

    print("[3/7] Enumerating ALL valid candidates per impulse and first availability...")
    availability, onset_detail = build_availability(impulses, candidates)
    print(
        f"      availability rows={len(availability):,} "
        f"onset associations={len(onset_detail):,}"
    )

    print("[4/7] Building redundancy, timing and 0s-vs-60s reports...")
    redundancy = redundancy_summary(availability)
    earliest = earliest_summary(availability)
    earliest_ordinal = earliest_by_ordinal(availability)
    stages = detection_stage_summary(availability)
    incremental = zero_vs_sixty_incremental(availability)
    spacing = spacing_buckets(candidates)
    day_profile, storm = day_and_storm_profiles(candidates)
    geometry = geometry_overlap(candidates)

    print("[5/7] Running invariants...")
    invariants = build_invariants(
        availability, onset_detail, candidates, impulses
    )
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    # Source identity hashes support deterministic reruns without hashing every
    # parquet byte (which can vary with metadata serialization).
    candidate_hashes = {}
    for policy in ALL_POLICIES:
        ids = sorted(
            candidates.loc[
                candidates["policy_id"].astype(str) == policy,
                "candidate_id",
            ].astype(str)
        )
        candidate_hashes[policy] = hashlib.sha256(
            "\n".join(ids).encode("utf-8")
        ).hexdigest()

    impulse_hash = hashlib.sha256(
        "\n".join(sorted(impulses["impulse_id"].astype(str))).encode("utf-8")
    ).hexdigest()

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "built_at_utc": utc_now_iso(),
        "frozen_phase7k2_builder_version": k2_meta.get("builder_version"),
        "frozen_phase7k3_builder_version": k3_meta.get("builder_version"),
        "primary_policy": PRIMARY_POLICY,
        "challenger_policy": CHALLENGER_POLICY,
        "targets_bps": list(TARGETS_BPS),
        "horizon_min": HORIZON_MIN,
        "common_impulse_count": int(len(impulses)),
        "common_reacceleration_count": reacc_n,
        "availability_rows": int(len(availability)),
        "onset_association_rows": int(len(onset_detail)),
        "candidate_id_set_sha256_by_policy": candidate_hashes,
        "common_impulse_id_set_sha256": impulse_hash,
        "scientific_state": "REDUNDANCY_EARLIEST_AVAILABILITY_AUDIT_NOT_YET_FROZEN",
        "failed_invariants": int(len(failed)),
    }

    run_config = {
        **summary,
        "onset_window_rule": (
            "first impulse: earliest common-parent anchor -> current 50 hit; "
            "reacceleration: previous common 50 hit -> current 50 hit"
        ),
        "freshness_rule": (
            "reacceleration requires previous 50 hit < candidate origin <= "
            "release when available <= candidate <= target hit"
        ),
        "redundancy_rule": (
            "2+ valid causal candidates inside the same common-50 onset window"
        ),
        "earliest_availability_rule": (
            "earliest valid causal candidate from onset-window start through target hit; "
            "no greedy one-to-one latest-candidate matching"
        ),
        "phase7k2_root": str(args.phase7k2_root),
        "phase7k3_root": str(args.phase7k3_root),
        "phase7i1_root": str(args.phase7i1_root),
    }

    findings = findings_text(
        status, redundancy, earliest, incremental, geometry, storm
    )

    print("[6/7] Writing atomic datasets...")
    atomic_write_parquet(
        availability,
        args.output_root / "impulse_candidate_availability.parquet",
    )
    atomic_write_parquet(
        onset_detail,
        args.output_root / "candidate_onset_associations.parquet",
    )
    atomic_write_json(
        args.output_root / "metadata" / "latest_summary.json", summary
    )
    atomic_write_json(
        args.output_root / "metadata" / "run_config.json", run_config
    )

    print("[7/7] Writing reports...")
    atomic_write_csv(redundancy, args.report_root / "candidate_redundancy_summary.csv")
    atomic_write_csv(earliest, args.report_root / "earliest_availability_summary.csv")
    atomic_write_csv(earliest_ordinal, args.report_root / "earliest_availability_by_ordinal.csv")
    atomic_write_csv(stages, args.report_root / "detection_stage_summary.csv")
    atomic_write_csv(incremental, args.report_root / "zero_vs_sixty_incremental_coverage.csv")
    atomic_write_csv(spacing, args.report_root / "candidate_spacing_buckets.csv")
    atomic_write_csv(day_profile, args.report_root / "candidate_day_profile.csv")
    atomic_write_csv(storm, args.report_root / "candidate_storm_windows.csv")
    atomic_write_csv(geometry, args.report_root / "zero_vs_sixty_geometry_overlap.csv")
    atomic_write_csv(invariants, args.report_root / "invariants.csv")
    atomic_write_text(args.report_root / "findings.md", findings)

    print("")
    print(f"Phase 7K.4 status: {status}")
    for policy in (PRIMARY_POLICY, CHALLENGER_POLICY):
        x = redundancy[
            (redundancy["policy_id"] == policy)
            & (redundancy["segment"] == "REACCEL")
        ]
        if len(x):
            r = x.iloc[0]
            print(
                f"{policy:22s} reaccel_onset={int(r['onset_covered_impulses']):4d}/"
                f"{int(r['impulses']):4d} redundant_share={100*safe_float(r['redundant_share_of_covered']):5.1f}% "
                f"extra_onset={int(r['extra_onset_candidates']):4d}"
            )
    for target in (75, 100):
        x = incremental[
            (incremental["target_bps"] == target)
            & (incremental["segment"] == "REACCEL")
        ]
        if len(x):
            r = x.iloc[0]
            print(
                f"{target:3d}bp reaccel: 0s-only={int(r['zero_only']):3d} "
                f"60s-only={int(r['sixty_only']):3d} "
                f"net={int(r['net_coverage_0s_minus_60s']):+3d}"
            )
    print(f"Failed invariants : {len(failed)}")
    print("Scientific state  : REDUNDANCY_EARLIEST_AVAILABILITY_AUDIT_NOT_YET_FROZEN")
    print(f"Reports           : {args.report_root}")

    if len(failed):
        print("")
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
