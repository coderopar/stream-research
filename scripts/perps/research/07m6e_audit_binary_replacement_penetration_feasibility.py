#!/usr/bin/env python3
"""Phase 7M.6E — binary replacement-penetration feasibility redesign.

Plain-English question
----------------------
Phase 7M.6D corrected the 3-second clock and showed that the continuous
replacement-consumption intensity is underpowered after matching on the full
causal opportunity set.  This phase asks a simpler primitive question:

    After opposing liquidity refills, did later aggressive execution
    demonstrably penetrate beyond the entire pre-refill residual queue at all?

The exposure is physically defined, not tuned:

    replacement_penetrated = replacement_execution_lower_bound_quote > 0

where Phase 7M.6D already computed the conservative lower bound from the
anchor-relative 3-second sequence.

Scientific safeguards
---------------------
* No raw market replay.
* No P75/P100/MFE/MAE/first-passage/continuation outcomes are read.
* No new outcome-driven threshold: zero is the physical queue-accounting bound.
* No composite signal and no ML.
* Exact candidate_date + direction matching is retained.
* The same z-caliper, Euclidean caliper, balance repair, projected-pair floors,
  and positive-share gates from frozen Phase 7M.6B are retained.
* Refill amount AND refill breadth remain primary controls; amount-only is a
  robustness diagnostic and cannot rescue a failed primary design.
* Binary exposure groups are matched directly (1 vs 0).  The continuous
  tercile matcher is intentionally NOT used because tied binary values would
  otherwise be split arbitrarily by rank.

Interpretation
--------------
PASS means the binary primitive has enough outcome-blind common support to
justify measuring it over the full candidate universe in a later replay.
FAIL means the observational design cannot cleanly separate replacement
penetration from the broader refill/high-turnover process under the frozen
controls; thresholds must not be loosened merely to force a test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
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
SCHEMA_NAME = "prospective_25bp_binary_replacement_penetration_feasibility_v1"
SOURCE_SCHEMA = "prospective_25bp_refill_redepletion_anchor_timing_audit_v1"
SOURCE_7M6B_SCHEMA = "prospective_25bp_refill_redepletion_sequence_v1"
PRIMARY_HORIZON_SEC = 3
EXPECTED_PILOT_DATES = 4
EXPECTED_PILOT_CANDIDATES = 446
EXPECTED_REFILL_ELIGIBLE = 443
TOTAL_CANDIDATE_DAYS = 62
DISCOVERY_END_DATE = "2026-06-17"
VALIDATION_START_DATE = "2026-06-18"
FULL_DISCOVERY_DAYS = 31
FULL_VALIDATION_DAYS = 31

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_7M6D_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SOURCE_SCHEMA
DEFAULT_7M6B_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6b_build_refill_redepletion_sequence_pilot.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_binary_replacement_penetration_feasibility"

# Only causal columns are read from 7M.6D.  The allow-list is the future-label firewall.
REQUIRED_SOURCE_COLUMNS = (
    "candidate_id", "candidate_date", "direction", "observation_horizon_sec",
    "timing_basis", "anchor_found", "anchor_ts_recv_ns", "anchor_offset_ms",
    "sequence_valid", "invalid_reason",
    "opposing__refilled_level_count", "opposing__observed_refill_quote",
    "opposing__strict_trade_time_complete_share",
    "opposing__replacement_execution_lower_bound_quote",
    "opposing__replacement_execution_lower_bound_to_refill_ratio",
    "control__directional_mid_move_bps_3s",
    "control__directional_aggressive_quote_3s",
    "control__opposing_peak_initial_queue_loss_share_3s",
    "control__opposing_refill_quote_3s_frozen",
    "is_reaccel_context",
)

FUTURE_NAME_TOKENS = (
    "outcome", "label", "p75", "p100", "mfe", "mae", "first_passage",
    "reaches_", "hit_", "continuation_target",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack() -> None:
    if pa is None or pq is None:
        raise SystemExit("Phase 7M.6E requires pandas, numpy and pyarrow in the project .venv.")


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6e-", suffix=path.suffix or ".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6e-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6e-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required existing phase script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_source(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    meta_path = root / "metadata" / "latest_summary.json"
    data_path = root / "corrected_anchor3s_candidate_metrics.parquet"
    if not meta_path.exists():
        raise RuntimeError(f"Missing Phase 7M.6D summary: {meta_path}")
    if not data_path.exists():
        raise RuntimeError(f"Missing corrected Phase 7M.6D candidate metrics: {data_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("schema_name")) != SOURCE_SCHEMA:
        raise RuntimeError(f"Unexpected Phase 7M.6D schema: {meta.get('schema_name')!r}")
    if bool(meta.get("future_outcomes_loaded")):
        raise RuntimeError("Phase 7M.6D source reports future outcomes loaded; refusing causal audit.")
    if bool(meta.get("raw_market_replay_performed")):
        raise RuntimeError("Phase 7M.6D source unexpectedly reports raw replay.")
    if str(meta.get("corrected_timing_basis")) != "anchor_ts_recv_ns + 3_000_000_000 ns":
        raise RuntimeError(f"Unexpected corrected timing basis: {meta.get('corrected_timing_basis')!r}")
    if not bool(meta.get("parity_direction_improved")):
        raise RuntimeError("Phase 7M.6D did not confirm that anchor-relative timing improved refill parity.")

    pf = pq.ParquetFile(data_path)
    names = set(pf.schema_arrow.names)
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in names]
    if missing:
        raise RuntimeError(f"Phase 7M.6D corrected metrics missing required causal columns: {missing}")
    suspicious = sorted(c for c in REQUIRED_SOURCE_COLUMNS if any(t in c.lower() for t in FUTURE_NAME_TOKENS))
    if suspicious:
        raise RuntimeError(f"Internal causal allow-list unexpectedly contains future-like names: {suspicious}")

    d = pq.read_table(data_path, columns=list(REQUIRED_SOURCE_COLUMNS)).to_pandas()
    d["candidate_id"] = d["candidate_id"].astype(str)
    d["candidate_date"] = d["candidate_date"].astype(str)
    if len(d) != EXPECTED_PILOT_CANDIDATES or d["candidate_id"].duplicated().any():
        raise RuntimeError(
            f"Expected {EXPECTED_PILOT_CANDIDATES} unique 7M.6D candidate rows; "
            f"found rows={len(d)}, unique={d['candidate_id'].nunique()}"
        )
    horizons = sorted(pd.to_numeric(d["observation_horizon_sec"], errors="coerce").dropna().astype(int).unique().tolist())
    if horizons != [PRIMARY_HORIZON_SEC]:
        raise RuntimeError(f"Unexpected corrected horizons: {horizons}")
    bases = sorted(d["timing_basis"].astype(str).unique().tolist())
    if bases != ["RECOVERED_ANCHOR_PLUS_3S"]:
        raise RuntimeError(f"Unexpected timing basis in corrected metrics: {bases}")
    if d["candidate_date"].nunique() != EXPECTED_PILOT_DATES:
        raise RuntimeError(f"Expected {EXPECTED_PILOT_DATES} pilot dates; found {d['candidate_date'].nunique()}")
    return meta, d


def make_analysis_frame(source: pd.DataFrame) -> pd.DataFrame:
    d = source.copy()
    d["__anchor"] = bool_series(d["anchor_found"])
    d["__valid"] = bool_series(d["sequence_valid"])
    d["__refill_quote"] = pd.to_numeric(d["opposing__observed_refill_quote"], errors="coerce")
    d["__time_complete"] = pd.to_numeric(d["opposing__strict_trade_time_complete_share"], errors="coerce")
    d["__lb_quote"] = pd.to_numeric(d["opposing__replacement_execution_lower_bound_quote"], errors="coerce")
    d["__refill_levels"] = pd.to_numeric(d["opposing__refilled_level_count"], errors="coerce")

    eligible = (
        d["__anchor"]
        & d["__valid"]
        & (d["__refill_quote"] > 0)
        & (d["__time_complete"] >= 1.0 - 1e-12)
        & d["__lb_quote"].notna()
        & d["__refill_levels"].notna()
    )
    d["replacement_penetration_eligible"] = eligible
    # Physical zero threshold only: positive conservative lower-bound quote means
    # executions exceeded the entire pre-refill residual at >=1 refilled level.
    d["replacement_penetrated"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    d.loc[eligible, "replacement_penetrated"] = (d.loc[eligible, "__lb_quote"] > 1e-12).astype(int).astype("Int64")

    d["__momentum"] = pd.to_numeric(d["control__directional_mid_move_bps_3s"], errors="coerce")
    d["__aggression"] = np.log1p(pd.to_numeric(d["control__directional_aggressive_quote_3s"], errors="coerce").clip(lower=0))
    d["__destruction"] = pd.to_numeric(d["control__opposing_peak_initial_queue_loss_share_3s"], errors="coerce")
    d["__sequence_refill"] = np.log1p(d["__refill_quote"].clip(lower=0))
    d["__refill_breadth"] = np.log1p(d["__refill_levels"].clip(lower=0))
    return d


def binary_greedy_pairs(
    d: pd.DataFrame,
    exposure: str,
    controls: Sequence[str],
    phase7m6b: Any,
) -> pd.DataFrame:
    """1:1 no-replacement matching of physically defined 1 vs 0 exposure."""
    x = d.copy()
    exp = pd.to_numeric(x[exposure], errors="coerce")
    if not exp.dropna().isin([0, 1]).all():
        bad = sorted(exp.dropna()[~exp.dropna().isin([0, 1])].unique().tolist())
        raise RuntimeError(f"Binary matcher received non-binary exposure values: {bad}")
    for c in controls:
        x[f"__z__{c}"] = phase7m6b.zscore(x[c])

    treated = x[exp == 1].copy()
    untreated = x[exp == 0].copy()
    rows: list[dict[str, Any]] = []

    for (dt, direction), hb in treated.groupby(["candidate_date", "direction"], sort=True):
        lb = untreated[
            (untreated["candidate_date"] == dt)
            & (pd.to_numeric(untreated["direction"], errors="coerce") == int(direction))
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
                    if (
                        not math.isfinite(hv)
                        or not math.isfinite(lv)
                        or abs(hv - lv) > float(phase7m6b.MATCH_Z_CALIPER)
                    ):
                        okay = False
                        break
                    diffs.append(hv - lv)
                if not okay:
                    continue
                dist = math.sqrt(sum(v * v for v in diffs))
                if dist > float(phase7m6b.MATCH_EUCLIDEAN_CALIPER):
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
                "penetrated_candidate_id": str(hr["candidate_id"]),
                "not_penetrated_candidate_id": str(lr["candidate_id"]),
                # compatibility aliases let the frozen balance/repair helpers run unchanged
                "high_candidate_id": str(hr["candidate_id"]),
                "low_candidate_id": str(lr["candidate_id"]),
                "distance": best[0][0],
                "high__replacement_penetrated": 1,
                "low__replacement_penetrated": 0,
            }
            for c, diff in zip(controls, diffs):
                row[f"high__{c}"] = safe_float(hr[c])
                row[f"low__{c}"] = safe_float(lr[c])
                row[f"diff__{c}"] = diff
            rows.append(row)
    return pd.DataFrame(rows)


def build_feasibility(d: pd.DataFrame, phase7m6b: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = d[bool_series(d["replacement_penetration_eligible"])].copy()
    required = [
        "replacement_penetrated", "__momentum", "__aggression", "__destruction",
        "__sequence_refill", "__refill_breadth",
    ]
    for c in required:
        z = z[pd.to_numeric(z[c], errors="coerce").notna()].copy()

    controls_by_name = {
        "REFILL_AMOUNT_BREADTH_PRIMARY": [
            "__momentum", "__aggression", "__destruction", "__sequence_refill", "__refill_breadth"
        ],
        "REFILL_AMOUNT_ONLY_ROBUSTNESS": [
            "__momentum", "__aggression", "__destruction", "__sequence_refill"
        ],
    }

    pilot_days = max(1, int(z["candidate_date"].nunique()))
    scale = TOTAL_CANDIDATE_DAYS / pilot_days
    disc_pilot_days = max(1, int(z[z["candidate_date"] <= DISCOVERY_END_DATE]["candidate_date"].nunique()))
    val_pilot_days = max(1, int(z[z["candidate_date"] >= VALIDATION_START_DATE]["candidate_date"].nunique()))
    positive_share = float((pd.to_numeric(z["replacement_penetrated"], errors="coerce") == 1).mean()) if len(z) else math.nan

    rows: list[dict[str, Any]] = []
    pair_parts: list[pd.DataFrame] = []
    for control_name, controls in controls_by_name.items():
        pairs = binary_greedy_pairs(z, "replacement_penetrated", controls, phase7m6b)
        repaired = phase7m6b.repair_pairs(pairs, controls) if len(pairs) else pairs
        mx, vals = phase7m6b.pair_balance(repaired, controls) if len(repaired) else (math.nan, {})
        projected = int(round(len(repaired) * scale))
        disc_pairs = int((repaired["candidate_date"].astype(str) <= DISCOVERY_END_DATE).sum()) if len(repaired) else 0
        val_pairs = int((repaired["candidate_date"].astype(str) >= VALIDATION_START_DATE).sum()) if len(repaired) else 0
        projected_disc = int(round(disc_pairs * FULL_DISCOVERY_DAYS / disc_pilot_days))
        projected_val = int(round(val_pairs * FULL_VALIDATION_DAYS / val_pilot_days))

        feasible = bool(
            len(repaired) > 0
            and math.isfinite(mx)
            and mx <= float(phase7m6b.BALANCE_SMD_GATE)
            and projected >= int(phase7m6b.MIN_PROJECTED_FULL_PRIMARY_PAIRS)
            and projected_disc >= int(phase7m6b.MIN_PROJECTED_SPLIT_PAIRS)
            and projected_val >= int(phase7m6b.MIN_PROJECTED_SPLIT_PAIRS)
            and math.isfinite(positive_share)
            and float(phase7m6b.MIN_FEATURE_POSITIVE_SHARE) <= positive_share <= float(phase7m6b.MAX_FEATURE_POSITIVE_SHARE)
        )
        row: dict[str, Any] = {
            "feature_role": "PRIMARY" if control_name == "REFILL_AMOUNT_BREADTH_PRIMARY" else "ROBUSTNESS",
            "feature": "replacement_penetrated",
            "feature_definition": "opposing__replacement_execution_lower_bound_quote > 0",
            "control_set": control_name,
            "pilot_eligible_candidates": int(len(z)),
            "pilot_penetrated_candidates": int((pd.to_numeric(z["replacement_penetrated"], errors="coerce") == 1).sum()),
            "pilot_not_penetrated_candidates": int((pd.to_numeric(z["replacement_penetrated"], errors="coerce") == 0).sum()),
            "pilot_days": pilot_days,
            "pilot_pairs_before_repair": int(len(pairs)),
            "pilot_pairs_after_repair": int(len(repaired)),
            "pilot_max_abs_smd": mx,
            "projected_full_pairs": projected,
            "projected_discovery_pairs": projected_disc,
            "projected_validation_pairs": projected_val,
            "positive_feature_share": positive_share,
            "balance_gate": float(phase7m6b.BALANCE_SMD_GATE),
            "projected_full_pair_gate": int(phase7m6b.MIN_PROJECTED_FULL_PRIMARY_PAIRS),
            "projected_split_pair_gate": int(phase7m6b.MIN_PROJECTED_SPLIT_PAIRS),
            "positive_share_min_gate": float(phase7m6b.MIN_FEATURE_POSITIVE_SHARE),
            "positive_share_max_gate": float(phase7m6b.MAX_FEATURE_POSITIVE_SHARE),
            "feasible_before_outcomes": feasible,
        }
        for c, v in vals.items():
            row[f"smd__{c}"] = v
        rows.append(row)
        if len(repaired):
            q = repaired.copy()
            q["control_set"] = control_name
            q["feature"] = "replacement_penetrated"
            pair_parts.append(q)

    return pd.DataFrame(rows), pd.concat(pair_parts, ignore_index=True, sort=False) if pair_parts else pd.DataFrame()


def subgroup_table(d: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    z = d[bool_series(d["replacement_penetration_eligible"])].copy()
    exp = pd.to_numeric(z["replacement_penetrated"], errors="coerce")
    primary_pairs = pairs[pairs["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY"].copy() if len(pairs) else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    definitions: list[tuple[str, str, pd.Series]] = [
        ("SPLIT", "DISCOVERY", z["candidate_date"] <= DISCOVERY_END_DATE),
        ("SPLIT", "VALIDATION", z["candidate_date"] >= VALIDATION_START_DATE),
        ("DIRECTION", "UP", pd.to_numeric(z["direction"], errors="coerce") > 0),
        ("DIRECTION", "DOWN", pd.to_numeric(z["direction"], errors="coerce") < 0),
        ("REACCEL", "REACCEL", bool_series(z["is_reaccel_context"])),
        ("REACCEL", "NON_REACCEL", ~bool_series(z["is_reaccel_context"])),
    ]
    for family, label, mask in definitions:
        g = z[mask].copy()
        ge = pd.to_numeric(g["replacement_penetrated"], errors="coerce")
        if len(primary_pairs):
            if family == "SPLIT" and label == "DISCOVERY":
                pg = primary_pairs[primary_pairs["candidate_date"] <= DISCOVERY_END_DATE]
            elif family == "SPLIT" and label == "VALIDATION":
                pg = primary_pairs[primary_pairs["candidate_date"] >= VALIDATION_START_DATE]
            elif family == "DIRECTION" and label == "UP":
                pg = primary_pairs[pd.to_numeric(primary_pairs["direction"], errors="coerce") > 0]
            elif family == "DIRECTION" and label == "DOWN":
                pg = primary_pairs[pd.to_numeric(primary_pairs["direction"], errors="coerce") < 0]
            elif family == "REACCEL":
                ids = set(g["candidate_id"].astype(str))
                pg = primary_pairs[
                    primary_pairs["penetrated_candidate_id"].astype(str).isin(ids)
                    & primary_pairs["not_penetrated_candidate_id"].astype(str).isin(ids)
                ]
            else:
                pg = primary_pairs.iloc[0:0]
        else:
            pg = pd.DataFrame()
        rows.append({
            "subgroup_family": family,
            "subgroup": label,
            "eligible_candidates": int(len(g)),
            "penetrated_candidates": int((ge == 1).sum()),
            "not_penetrated_candidates": int((ge == 0).sum()),
            "penetrated_share": float((ge == 1).mean()) if len(g) else math.nan,
            "primary_matched_pairs": int(len(pg)),
            "both_exposure_states_present": bool((ge == 1).any() and (ge == 0).any()),
        })
    return pd.DataFrame(rows)


def findings_text(summary: dict[str, Any], gate: pd.DataFrame, subgroups: pd.DataFrame) -> str:
    p = gate[gate["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY"]
    r = gate[gate["control_set"] == "REFILL_AMOUNT_ONLY_ROBUSTNESS"]
    pr = p.iloc[0].to_dict() if len(p) else {}
    rr = r.iloc[0].to_dict() if len(r) else {}
    lines = [
        "# Phase 7M.6E — Binary replacement-penetration feasibility redesign",
        "",
        f"- Status: **{summary['status']}**",
        f"- Pilot dates: **{', '.join(summary['processed_dates'])}**",
        f"- Corrected anchor+3s candidates: **{summary['candidate_count']}**",
        f"- Penetration-eligible candidates: **{summary['eligible_candidate_count']}**",
        "- Future outcomes loaded: **NO**",
        "- Raw market replay performed: **NO**",
        "",
        "## Primitive exposure",
        "",
        "`replacement_penetrated = opposing__replacement_execution_lower_bound_quote > 0`",
        "",
        "A positive state means aggressive execution exceeded the entire pre-refill residual queue at at least one refilled opposing price, under the conservative anchor-relative 3-second attribution. Zero is a physical accounting boundary, not a tuned predictive threshold.",
        "",
        "## Primary common-support gate",
        "",
        "Controls: exact candidate_date + direction, momentum, aggressive flow, initial opposing destruction, anchor-relative refill amount, and refill breadth.",
    ]
    if pr:
        lines += [
            f"- Penetrated / not penetrated: **{int(pr.get('pilot_penetrated_candidates', 0))} / {int(pr.get('pilot_not_penetrated_candidates', 0))}**",
            f"- Penetrated share: **{safe_float(pr.get('positive_feature_share')):.2%}**",
            f"- Pilot pairs before / after repair: **{int(pr.get('pilot_pairs_before_repair', 0))} / {int(pr.get('pilot_pairs_after_repair', 0))}**",
            f"- Max |SMD|: **{safe_float(pr.get('pilot_max_abs_smd')):.3f}**",
            f"- Projected full pairs: **{int(pr.get('projected_full_pairs', 0))}**",
            f"- Projected discovery / validation pairs: **{int(pr.get('projected_discovery_pairs', 0))} / {int(pr.get('projected_validation_pairs', 0))}**",
            f"- Feasible before outcomes: **{bool(pr.get('feasible_before_outcomes'))}**",
        ]
    lines += ["", "## Amount-only robustness", ""]
    if rr:
        lines += [
            f"- Pilot pairs after repair: **{int(rr.get('pilot_pairs_after_repair', 0))}**",
            f"- Max |SMD|: **{safe_float(rr.get('pilot_max_abs_smd')):.3f}**",
            f"- Projected full pairs: **{int(rr.get('projected_full_pairs', 0))}**",
            f"- Projected discovery / validation pairs: **{int(rr.get('projected_discovery_pairs', 0))} / {int(rr.get('projected_validation_pairs', 0))}**",
            f"- Feasible before outcomes: **{bool(rr.get('feasible_before_outcomes'))}**",
        ]
    lines += [
        "",
        "Amount-only robustness cannot rescue a failure of the primary amount+breadth design because refill breadth was already independently implicated in Phase 7M.5.",
        "",
        "## Subgroup representation",
        "",
    ]
    for row in subgroups.itertuples(index=False):
        lines.append(
            f"- {row.subgroup_family}/{row.subgroup}: eligible={int(row.eligible_candidates)}, "
            f"penetrated={int(row.penetrated_candidates)}, not_penetrated={int(row.not_penetrated_candidates)}, "
            f"primary_pairs={int(row.primary_matched_pairs)}"
        )
    lines += [
        "",
        "No new subgroup numeric threshold is invented here. Discovery/validation pair floors remain the preregistered hard split gate; direction and reacceleration counts are reported outcome-blind for representation diagnostics.",
        "",
        "## Decision rule",
        "",
        "Only a PASS of the primary REFILL_AMOUNT_BREADTH_PRIMARY gate justifies a later full replay for this primitive. If it fails, do not loosen calipers, remove refill breadth, or change the physical zero boundary after seeing the result.",
    ]
    return "\n".join(lines) + "\n"


def run_self_test(phase7m6b_script: Path) -> int:
    phase7m6b = load_module(phase7m6b_script, "phase7m6b_for_7m6e_selftest")
    rows = []
    # Balanced binary exposure in two exact date/direction cells with near-identical controls.
    for day in ("2026-06-01", "2026-06-20"):
        for direction in (-1, 1):
            for i in range(24):
                rows.append({
                    "candidate_id": f"{day}_{direction}_{i}",
                    "candidate_date": day,
                    "direction": direction,
                    "replacement_penetrated": i % 2,
                    "a": float(i // 2),
                    "b": float((i // 2) % 4),
                    "c": float((i // 2) % 3),
                    "d": float((i // 2) % 5),
                    "e": float((i // 2) % 6),
                })
    d = pd.DataFrame(rows)
    pairs = binary_greedy_pairs(d, "replacement_penetrated", ["a", "b", "c", "d", "e"], phase7m6b)
    assert len(pairs) > 0
    assert (pd.to_numeric(pairs["high__replacement_penetrated"], errors="coerce") == 1).all()
    assert (pd.to_numeric(pairs["low__replacement_penetrated"], errors="coerce") == 0).all()
    assert (pairs["penetrated_candidate_id"].astype(str) != pairs["not_penetrated_candidate_id"].astype(str)).all()

    # Physical boundary: exactly zero is not penetrated; any positive conservative lower bound is penetrated.
    fake = pd.DataFrame({
        "candidate_id": ["z", "p"], "candidate_date": ["2026-06-01"] * 2, "direction": [1, 1],
        "observation_horizon_sec": [3, 3], "timing_basis": ["RECOVERED_ANCHOR_PLUS_3S"] * 2,
        "anchor_found": [True, True], "anchor_ts_recv_ns": [1, 2], "anchor_offset_ms": [900.0, 900.0],
        "sequence_valid": [True, True], "invalid_reason": ["", ""],
        "opposing__refilled_level_count": [2, 2], "opposing__observed_refill_quote": [1000.0, 1000.0],
        "opposing__strict_trade_time_complete_share": [1.0, 1.0],
        "opposing__replacement_execution_lower_bound_quote": [0.0, 0.01],
        "opposing__replacement_execution_lower_bound_to_refill_ratio": [0.0, 0.00001],
        "control__directional_mid_move_bps_3s": [1.0, 1.0],
        "control__directional_aggressive_quote_3s": [10.0, 10.0],
        "control__opposing_peak_initial_queue_loss_share_3s": [0.5, 0.5],
        "control__opposing_refill_quote_3s_frozen": [1000.0, 1000.0],
        "is_reaccel_context": [False, False],
    })
    a = make_analysis_frame(fake)
    assert int(a.loc[0, "replacement_penetrated"]) == 0
    assert int(a.loc[1, "replacement_penetrated"]) == 1
    print("PASS: Phase 7M.6E self-test (physical binary exposure + direct 1-vs-0 frozen-caliper matcher).")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7M.6E binary replacement-penetration feasibility redesign")
    p.add_argument("--phase7m6d-root", type=Path, default=DEFAULT_7M6D_ROOT)
    p.add_argument("--phase7m6b-script", type=Path, default=DEFAULT_7M6B_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args.phase7m6b_script)
    require_stack()

    print("Phase 7M.6E — Binary replacement-penetration feasibility redesign")
    print("[1/6] Verifying corrected Phase 7M.6D anchor+3s causal source...")
    source_meta, source = verify_source(args.phase7m6d_root)
    phase7m6b = load_module(args.phase7m6b_script, "phase7m6b_for_7m6e")

    # Hard compatibility with preregistered 7M.6B gates/calipers.
    expected_constants = {
        "MATCH_Z_CALIPER": 0.75,
        "MATCH_EUCLIDEAN_CALIPER": 1.00,
        "BALANCE_SMD_GATE": 0.20,
        "MIN_PROJECTED_FULL_PRIMARY_PAIRS": 250,
        "MIN_PROJECTED_SPLIT_PAIRS": 100,
        "MIN_FEATURE_POSITIVE_SHARE": 0.10,
        "MAX_FEATURE_POSITIVE_SHARE": 0.90,
    }
    for name, expected in expected_constants.items():
        actual = safe_float(getattr(phase7m6b, name, math.nan))
        if not math.isfinite(actual) or abs(actual - float(expected)) > 1e-12:
            raise RuntimeError(f"Frozen Phase 7M.6B constant mismatch for {name}: expected {expected}, found {actual}")

    print("[2/6] Defining the physical binary penetration state (no tuned threshold)...")
    analysis = make_analysis_frame(source)
    eligible = analysis[bool_series(analysis["replacement_penetration_eligible"])].copy()
    if len(eligible) != EXPECTED_REFILL_ELIGIBLE:
        raise RuntimeError(f"Expected {EXPECTED_REFILL_ELIGIBLE} penetration-eligible candidates from 7M.6D; found {len(eligible)}")
    exp = pd.to_numeric(eligible["replacement_penetrated"], errors="coerce")
    if not exp.isin([0, 1]).all():
        raise RuntimeError("Eligible binary exposure contains missing/non-binary values.")

    print("[3/6] Matching penetrated vs not-penetrated under frozen causal controls...")
    gate, pairs = build_feasibility(analysis, phase7m6b)
    primary = gate[gate["control_set"] == "REFILL_AMOUNT_BREADTH_PRIMARY"]
    if len(primary) != 1:
        raise RuntimeError("Primary binary feasibility row missing or duplicated.")
    primary_feasible = bool(primary.iloc[0]["feasible_before_outcomes"])

    print("[4/6] Auditing discovery/validation, direction and reacceleration representation...")
    subgroups = subgroup_table(analysis, pairs)

    print("[5/6] Writing causal manifests and invariants...")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    manifest_cols = [
        "candidate_id", "candidate_date", "direction", "anchor_found", "sequence_valid",
        "opposing__observed_refill_quote", "opposing__refilled_level_count",
        "opposing__replacement_execution_lower_bound_quote",
        "replacement_penetration_eligible", "replacement_penetrated",
        "control__directional_mid_move_bps_3s", "control__directional_aggressive_quote_3s",
        "control__opposing_peak_initial_queue_loss_share_3s", "is_reaccel_context",
    ]
    atomic_parquet(args.output_root / "binary_penetration_candidate_manifest.parquet", analysis[manifest_cols].copy())
    atomic_csv(args.report_root / "binary_penetration_precompute_gate.csv", gate)
    atomic_csv(args.report_root / "binary_penetration_subgroup_feasibility.csv", subgroups)
    atomic_csv(args.report_root / "binary_penetration_pair_manifest.csv", pairs)

    all_subgroups_both = bool(subgroups["both_exposure_states_present"].all()) if len(subgroups) else False
    inv = pd.DataFrame([
        {"invariant": "future_outcomes_loaded", "pass": True, "actual": False, "required": False},
        {"invariant": "raw_market_replay_performed", "pass": True, "actual": False, "required": False},
        {"invariant": "source_is_corrected_anchor_plus_3s", "pass": True, "actual": source_meta.get("corrected_timing_basis"), "required": "anchor_ts_recv_ns + 3_000_000_000 ns"},
        {"invariant": "pilot_candidate_count", "pass": len(analysis) == EXPECTED_PILOT_CANDIDATES, "actual": len(analysis), "required": EXPECTED_PILOT_CANDIDATES},
        {"invariant": "penetration_eligible_count", "pass": len(eligible) == EXPECTED_REFILL_ELIGIBLE, "actual": len(eligible), "required": EXPECTED_REFILL_ELIGIBLE},
        {"invariant": "binary_exposure_only", "pass": bool(exp.isin([0, 1]).all()), "actual": sorted(exp.unique().tolist()), "required": [0, 1]},
        {"invariant": "physical_zero_boundary_only", "pass": True, "actual": "lower_bound_quote > 0", "required": "no tuned threshold"},
        {"invariant": "tercile_matcher_not_used", "pass": True, "actual": "direct 1 vs 0", "required": "direct binary groups"},
        {"invariant": "matching_calipers_and_gates_unchanged", "pass": True, "actual": expected_constants, "required": "7M.6B frozen values"},
        {"invariant": "refill_breadth_retained_in_primary_controls", "pass": True, "actual": "__refill_breadth", "required": True},
        {"invariant": "all_key_subgroups_have_both_exposure_states", "pass": all_subgroups_both, "actual": all_subgroups_both, "required": True},
        {"invariant": "primary_common_support_feasible_before_outcomes", "pass": primary_feasible, "actual": primary_feasible, "required": True},
    ])
    atomic_csv(args.report_root / "analysis_invariants.csv", inv)

    robustness = gate[gate["control_set"] == "REFILL_AMOUNT_ONLY_ROBUSTNESS"]
    robustness_feasible = bool(robustness.iloc[0]["feasible_before_outcomes"]) if len(robustness) else False
    if primary_feasible:
        status = "BINARY_PENETRATION_FEASIBLE_FOR_FULL_REPLAY"
    elif robustness_feasible:
        status = "PRIMARY_FAILS_AMOUNT_ONLY_PASSES_DO_NOT_PROMOTE"
    else:
        status = "BINARY_PENETRATION_UNDERPOWERED_UNDER_FROZEN_CONTROLS"

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "source_schema": SOURCE_SCHEMA,
        "source_builder_version": source_meta.get("builder_version"),
        "processed_dates": sorted(analysis["candidate_date"].unique().tolist()),
        "candidate_count": int(len(analysis)),
        "eligible_candidate_count": int(len(eligible)),
        "penetrated_candidate_count": int((exp == 1).sum()),
        "not_penetrated_candidate_count": int((exp == 0).sum()),
        "penetrated_share": float((exp == 1).mean()),
        "future_outcomes_loaded": False,
        "raw_market_replay_performed": False,
        "timing_basis": "RECOVERED_ANCHOR_PLUS_3S",
        "binary_exposure": "opposing__replacement_execution_lower_bound_quote > 0",
        "primary_controls": ["momentum", "aggression", "initial_destruction", "sequence_refill_amount", "refill_breadth"],
        "primary_common_support_feasible_before_outcomes": primary_feasible,
        "amount_only_robustness_feasible_before_outcomes": robustness_feasible,
        "full_replay_authorized_by_this_gate": primary_feasible,
        "future_action_if_fail": "Do not relax calipers/remove refill breadth/change binary zero boundary to force feasibility.",
    }
    atomic_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_json(args.output_root / "metadata" / "run_config.json", {
        "generated_at": utc_now_iso(),
        "source_root": str(args.phase7m6d_root),
        "phase7m6b_script": str(args.phase7m6b_script),
        "future_outcomes_loaded": False,
        "raw_market_replay_performed": False,
        "binary_exposure_definition": "replacement_execution_lower_bound_quote > 0",
        "matching": "direct binary 1:1 no replacement, exact date+direction, frozen z/euclidean calipers, frozen SMD repair",
        "primary_control_set": "momentum + aggression + initial destruction + anchor-relative refill amount + refill breadth",
    })
    atomic_text(args.report_root / "findings.md", findings_text(summary, gate, subgroups))

    print("[6/6] Complete.")
    p = primary.iloc[0]
    print(f"Audit status            : {status}")
    print(f"Eligible candidates     : {len(eligible):,}")
    print(f"Penetrated / not        : {int((exp == 1).sum()):,} / {int((exp == 0).sum()):,}")
    print(f"Penetrated share        : {float((exp == 1).mean()):.2%}")
    print(f"Primary pairs before    : {int(p['pilot_pairs_before_repair']):,}")
    print(f"Primary pairs after     : {int(p['pilot_pairs_after_repair']):,}")
    print(f"Primary max |SMD|       : {safe_float(p['pilot_max_abs_smd']):.3f}")
    print(f"Projected full pairs    : {int(p['projected_full_pairs']):,}")
    print(f"Projected D / V pairs   : {int(p['projected_discovery_pairs']):,} / {int(p['projected_validation_pairs']):,}")
    print(f"Primary feasible        : {primary_feasible}")
    print(f"Reports                 : {args.report_root}")
    print("Future outcomes loaded  : NO")
    print("Raw market replay       : NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
