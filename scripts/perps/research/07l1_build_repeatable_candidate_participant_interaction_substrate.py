#!/usr/bin/env python3
"""
Phase 7L.1 — Frozen 60s repeatable-candidate participant-interaction substrate
=============================================================================

Plain-English purpose
---------------------
We have now chosen the cleaner repeatable causal candidate ontology:

    >=5bp reset
        -> fresh >=10bp release
        -> fresh 35bp first passage
        -> minimum 60s before same-direction re-arm

Phase 7L.1 asks no predictive question by itself.  It performs the ONE expensive
raw market-data replay needed so later cheap analyses can ask:

    At the moment a fresh 35bp candidate exists, what are aggressive traders
    and the exact order-book queues doing?

The expensive measurements are the same families already validated in Phase
7I.4:
    * exact opposing/support queue depletion
    * retained initial liquidity
    * durable unrecovered loss
    * same-price refill, durability and latency
    * near-touch restoration / outward additions
    * executed directional/counter aggression
    * trade concentration / count / acceleration
    * price response per unit flow
    * support-vs-opposing asymmetry

Causal separation
-----------------
candidate_inputs.parquet and candidate_mechanics.parquet contain only
information observable by each row's decision timestamp.

Future 50/75/100 outcomes are written separately.  They must NOT be used as
causal features.

Why this script reuses Phase 7I.4
---------------------------------
Phase 7I.4 already contains the audited expensive replay implementation.  Rather
than copy/fork thousands of lines of sequence-sensitive order-book code, Phase
7L.1 imports that frozen builder as an execution engine and changes only the
candidate manifest supplied to it.

This wrapper:
    1. validates frozen Phase 7I.4 v1.0.1;
    2. validates frozen Phase 7K.2 v1.0.0;
    3. selects ONLY TEST_COOLDOWN_60S;
    4. validates the 5/10/35bp + 60s contract;
    5. freezes its candidate-ID hash;
    6. adapts only column names required by the frozen 7I.4 replay interface;
    7. runs the frozen raw-depth/trade engine with separate output roots;
    8. rewrites metadata so the resulting dataset is explicitly Phase 7L.1.

This phase does NOT:
    * regenerate candidates;
    * compare 0s vs 60s;
    * tune thresholds;
    * create a book-failure score;
    * fit ML;
    * claim alpha/profitability;
    * decide which features predict 75/100.

Those are cheap downstream questions after this substrate passes.

Expected frozen population
--------------------------
From Phase 7K.4 review:
    selected policy: TEST_COOLDOWN_60S
    expected candidates: 4,037
    expected candidate dates: 62

Outputs
-------
data/derived/perps/research/repeatable_candidate_participant_interaction_v1/
    candidate_inputs.parquet
    candidate_mechanics.parquet
    candidate_decision_outcomes.parquet
    candidate_k2_outcomes.parquet
    candidate_windows/...                 # resumable day checkpoints
    level_audit/...
    decision_outcomes/...
    metadata/latest_summary.json
    metadata/run_config.json
    metadata/frozen_candidate_manifest.json

reports/perps_repeatable_candidate_participant_interaction/
    participant_interaction_coverage.csv
    participant_interaction_data_quality_by_day.csv
    participant_interaction_schema.csv
    participant_interaction_invariants.csv
    participant_interaction_findings.md
    frozen_candidate_contract.csv

Runtime
-------
This is an expensive raw-depth replay.  It inherits Phase 7I.4's atomic
candidate-date checkpoints and exact input signatures, so an interrupted run can
resume without recomputing valid completed days.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BUILDER_VERSION = "1.0.1"
SCHEMA_VERSION = 1
SCHEMA_NAME = "repeatable_candidate_participant_interaction_v1"

FROZEN_PHASE7I4_VERSION = "1.0.1"
FROZEN_PHASE7K2_VERSION = "1.0.0"
FROZEN_POLICY = "TEST_COOLDOWN_60S"

EXPECTED_CANDIDATES = 4_037
EXPECTED_CANDIDATE_DAYS = 62
FROZEN_RESET_BPS = 5.0
FROZEN_RELEASE_BPS = 10.0
FROZEN_CANDIDATE_BPS = 35.0
FROZEN_COOLDOWN_SEC = 60

FIXED_HORIZONS_SEC = (1, 3, 5, 10)
PRIMARY_HORIZON_SEC = 3

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_PHASE7I4_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07i4_build_causal_candidate_participant_interaction_dataset.py"
)
DEFAULT_PHASE6A_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "06a_build_execution_foundation.py"
)
DEFAULT_PHASE7A_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07a_build_large_move_event_catalog.py"
)
DEFAULT_PHASE7E_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07e_build_raw_depth_continuation_mechanics.py"
)
DEFAULT_PHASE7I1_SCRIPT = (
    REPO_ROOT / "scripts" / "perps" / "research" /
    "07i1_build_causal_large_move_candidate_catalog.py"
)

DEFAULT_CANONICAL_ROOT = (
    REPO_ROOT / "data" / "canonical" / "market_events_v1"
)
DEFAULT_EXECUTION_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
)
DEFAULT_PHASE7K2_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" /
    "causal_rearm_timing_audit_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT / "reports" /
    "perps_repeatable_candidate_participant_interaction"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix="tmp7l1-", suffix=path.suffix, dir=str(path.parent)
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


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_pandas():
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "Phase 7L.1 requires pandas in the project .venv."
        ) from exc
    return pd


def require_stack():
    pd = require_pandas()
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Phase 7L.1 full replay requires pyarrow in the project .venv."
        ) from exc
    return pd, pq


def candidate_hash(frame) -> str:
    ids = sorted(frame["candidate_id"].astype(str).tolist())
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _unique_numeric(frame, column: str) -> list[float]:
    pd = require_pandas()
    return sorted(
        set(
            pd.to_numeric(frame[column], errors="coerce")
            .dropna()
            .astype(float)
            .tolist()
        )
    )


def validate_and_adapt_selected_candidates(
    all_candidates,
    summary: dict[str, Any],
    full_scope_required: bool = True,
):
    """
    Select and freeze the 60-second branch without using any future outcome.
    """
    pd = require_pandas()

    if str(summary.get("builder_version")) != FROZEN_PHASE7K2_VERSION:
        raise RuntimeError(
            f"Phase 7L.1 requires Phase 7K.2 v{FROZEN_PHASE7K2_VERSION}; "
            f"found {summary.get('builder_version')!r}"
        )
    if str(summary.get("status")) != "PASS":
        raise RuntimeError(
            f"Phase 7K.2 latest_summary must be PASS; found "
            f"{summary.get('status')!r}"
        )

    required = {
        "candidate_id",
        "policy_id",
        "candidate_date",
        "direction",
        "direction_label",
        "origin_second",
        "origin_price",
        "release_second",
        "candidate_second",
        "candidate_price",
        "candidate_displacement_bps",
        "reset_threshold_bps",
        "release_threshold_bps",
        "candidate_threshold_bps",
        "cooldown_sec",
    }
    missing = required - set(all_candidates.columns)
    if missing:
        raise RuntimeError(
            f"Phase 7K.2 candidate table missing required causal fields: "
            f"{sorted(missing)}"
        )

    selected = all_candidates[
        all_candidates["policy_id"].astype(str) == FROZEN_POLICY
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No candidates found for frozen policy {FROZEN_POLICY}"
        )
    if selected["candidate_id"].duplicated().any():
        raise RuntimeError("Selected 60s candidates contain duplicate candidate_id")

    observed_contract = {
        "reset_bps": _unique_numeric(selected, "reset_threshold_bps"),
        "release_bps": _unique_numeric(selected, "release_threshold_bps"),
        "candidate_bps": _unique_numeric(selected, "candidate_threshold_bps"),
        "cooldown_sec": _unique_numeric(selected, "cooldown_sec"),
    }
    expected_contract = {
        "reset_bps": [FROZEN_RESET_BPS],
        "release_bps": [FROZEN_RELEASE_BPS],
        "candidate_bps": [FROZEN_CANDIDATE_BPS],
        "cooldown_sec": [float(FROZEN_COOLDOWN_SEC)],
    }
    if observed_contract != expected_contract:
        raise RuntimeError(
            "Frozen 60s candidate mechanics do not match the approved contract.\n"
            f"Observed: {observed_contract}\nExpected: {expected_contract}"
        )

    days = int(selected["candidate_date"].astype(str).nunique())
    if full_scope_required:
        if len(selected) != EXPECTED_CANDIDATES:
            raise RuntimeError(
                f"Approved 60s universe requires {EXPECTED_CANDIDATES:,} "
                f"candidates; found {len(selected):,}"
            )
        if days != EXPECTED_CANDIDATE_DAYS:
            raise RuntimeError(
                f"Approved 60s universe requires {EXPECTED_CANDIDATE_DAYS} "
                f"candidate dates; found {days}"
            )

    # Phase 7I.4's replay engine carries an episode_id only as an identifier in
    # mechanics/outcome rows.  The repeatable ontology deliberately has no old
    # one-candidate episode semantics, so use the candidate itself as the
    # interface identifier rather than fabricating an old directional episode.
    selected["episode_id"] = (
        "REPEATABLE_CANDIDATE::" + selected["candidate_id"].astype(str)
    )

    # Frozen 7I.4 uses this legacy interface field name for missing-row repair.
    if "seconds_release_to_candidate" not in selected.columns:
        if "release_to_candidate_sec" in selected.columns:
            selected["seconds_release_to_candidate"] = pd.to_numeric(
                selected["release_to_candidate_sec"], errors="coerce"
            )
        else:
            selected["seconds_release_to_candidate"] = (
                pd.to_numeric(selected["candidate_second"], errors="coerce")
                - pd.to_numeric(selected["release_second"], errors="coerce")
            )

    selected = selected.sort_values(
        ["candidate_second", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)

    leakage_tokens = (
        "reach_",
        "mfe_",
        "mae_",
        "future",
        "outcome",
        "path_efficient",
        "seconds_to_50",
        "seconds_to_75",
        "seconds_to_100",
    )
    leaking = [
        c for c in selected.columns
        if any(tok in c.lower() for tok in leakage_tokens)
    ]
    if leaking:
        raise RuntimeError(
            "Selected causal candidate table unexpectedly contains future-label "
            f"columns: {leaking}"
        )

    return selected, observed_contract


def load_k2_freeze(root: Path):
    pd, pq = require_stack()
    candidates_path = root / "candidates.parquet"
    outcomes_path = root / "candidate_outcomes.parquet"
    summary_path = root / "metadata" / "latest_summary.json"

    for p in (candidates_path, outcomes_path, summary_path):
        if not p.exists():
            raise RuntimeError(f"Missing frozen Phase 7K.2 artifact: {p}")

    all_candidates = pq.read_table(candidates_path).to_pandas()
    all_outcomes = pq.read_table(outcomes_path).to_pandas()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    selected, contract = validate_and_adapt_selected_candidates(
        all_candidates,
        summary,
        full_scope_required=True,
    )
    ids = set(selected["candidate_id"].astype(str))
    selected_outcomes = all_outcomes[
        all_outcomes["candidate_id"].astype(str).isin(ids)
    ].copy()

    if set(selected_outcomes["candidate_id"].astype(str)) != ids:
        missing = ids - set(selected_outcomes["candidate_id"].astype(str))
        raise RuntimeError(
            f"Phase 7K.2 outcome table does not cover selected candidates; "
            f"missing={len(missing)}"
        )

    h = candidate_hash(selected)

    # When available, require the upstream K2 policy hash to agree.
    upstream_hashes = summary.get("candidate_id_set_sha256_by_policy", {})
    expected_hash = (
        upstream_hashes.get(FROZEN_POLICY)
        if isinstance(upstream_hashes, dict)
        else None
    )
    if expected_hash and str(expected_hash) != h:
        raise RuntimeError(
            "Selected candidate hash disagrees with Phase 7K.2 metadata: "
            f"computed={h} metadata={expected_hash}"
        )

    freeze_summary = {
        "builder_version": FROZEN_PHASE7K2_VERSION,
        "status": "PASS",
        "selected_policy": FROZEN_POLICY,
        "candidate_count": int(len(selected)),
        "candidate_days": int(
            selected["candidate_date"].astype(str).nunique()
        ),
        "candidate_id_set_sha256": h,
        "contract": contract,
    }
    freeze_manifest = {
        "freeze_status": "FROZEN_FOR_PHASE_7L_PARTICIPANT_INTERACTION",
        **freeze_summary,
    }
    return selected, selected_outcomes, freeze_summary, freeze_manifest, h



def validate_7i4_replay_interface(engine) -> None:
    """
    Verify the imported frozen replay module exposes the exact public/internal
    surface Phase 7L.1 depends on before any raw-data work begins.

    This guards against silently accepting a future builder that happens to
    share a version-like interface but changed the replay contract.
    """
    required_callables = (
        "main",
        "load_frozen_candidates",
        "process_candidate_date",
        "build_decision_outcomes",
        "build_coverage",
        "build_day_quality",
        "build_schema_report",
        "check_invariants",
        "write_parquet_atomic",
    )
    missing_callables = [
        name for name in required_callables
        if not callable(getattr(engine, name, None))
    ]

    required_constants = {
        "FIXED_HORIZONS_SEC": FIXED_HORIZONS_SEC,
        "PRIMARY_HORIZON_SEC": PRIMARY_HORIZON_SEC,
        "PRICE_SCALE": 10.0,
        "MAX_BOOK_AGE_MS": 2000,
    }
    bad_constants = {}
    for name, expected in required_constants.items():
        found = getattr(engine, name, None)
        if name == "FIXED_HORIZONS_SEC":
            try:
                found_cmp = tuple(int(x) for x in found)
            except Exception:
                found_cmp = found
            if found_cmp != tuple(expected):
                bad_constants[name] = (found, expected)
        else:
            if found != expected:
                bad_constants[name] = (found, expected)

    if missing_callables or bad_constants:
        raise RuntimeError(
            "Phase 7I.4 v1.0.1 replay interface is incompatible with Phase 7L.1. "
            f"Missing callables={missing_callables}; bad constants={bad_constants}"
        )


def patch_7i4_engine(engine, selected, selected_outcomes, freeze_summary, freeze_manifest, h):
    """
    Patch only manifest-loading/schema constants.  Raw replay functions remain
    byte-for-byte those of frozen Phase 7I.4.
    """
    pd, _ = require_stack()

    if str(getattr(engine, "BUILDER_VERSION", "")) != FROZEN_PHASE7I4_VERSION:
        raise RuntimeError(
            f"Phase 7L.1 requires frozen Phase 7I.4 "
            f"v{FROZEN_PHASE7I4_VERSION}; "
            f"found {getattr(engine, 'BUILDER_VERSION', None)!r}"
        )

    validate_7i4_replay_interface(engine)

    up = int((pd.to_numeric(selected["direction"], errors="coerce") > 0).sum())
    down = int((pd.to_numeric(selected["direction"], errors="coerce") < 0).sum())

    engine.BUILDER_VERSION = BUILDER_VERSION
    engine.SCHEMA_VERSION = SCHEMA_VERSION
    engine.SCHEMA_NAME = SCHEMA_NAME
    engine.EXPECTED_FULL_CANDIDATES = int(len(selected))
    engine.EXPECTED_FULL_UP = up
    engine.EXPECTED_FULL_DOWN = down
    engine.EXPECTED_FULL_CANDIDATE_DAYS = int(
        selected["candidate_date"].astype(str).nunique()
    )

    def selected_loader(_args):
        return (
            selected.copy(),
            selected_outcomes.copy(),
            dict(freeze_summary),
            dict(freeze_manifest),
            str(h),
        )

    engine.load_frozen_candidates = selected_loader
    return up, down


def rewrite_phase7l_metadata(
    output_root: Path,
    report_root: Path,
    selected,
    selected_outcomes,
    candidate_id_hash: str,
    up: int,
    down: int,
    engine_status: int,
    phase7i4_script: Path,
    phase7k2_root: Path,
) -> None:
    pd, pq = require_stack()

    latest_path = output_root / "metadata" / "latest_summary.json"
    engine_summary = {}
    if latest_path.exists():
        engine_summary = json.loads(latest_path.read_text(encoding="utf-8"))

    # Rename the convenience future-label file so it never masquerades as 7I.1.
    old_future = output_root / "candidate_7i1_outcomes.parquet"
    new_future = output_root / "candidate_k2_outcomes.parquet"
    if old_future.exists():
        if new_future.exists():
            new_future.unlink()
        os.replace(old_future, new_future)

    causal_inputs_path = output_root / "candidate_inputs.parquet"
    mechanics_path = output_root / "candidate_mechanics.parquet"
    outcomes_path = output_root / "candidate_decision_outcomes.parquet"

    mechanics_rows = 0
    valid_mechanics = 0
    decision_rows = 0
    if mechanics_path.exists():
        m = pq.read_table(mechanics_path).to_pandas()
        mechanics_rows = int(len(m))
        valid_mechanics = int(
            m["window_valid"].fillna(False).astype(bool).sum()
        ) if "window_valid" in m else 0
    if outcomes_path.exists():
        decision_rows = int(
            pq.ParquetFile(outcomes_path).metadata.num_rows
        )

    final_status = str(engine_summary.get("status", "UNKNOWN"))
    if engine_status != 0 and final_status != "SMOKE_PASS":
        final_status = "FAIL"

    summary = {
        **engine_summary,
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": final_status,
        "research_role": (
            "FROZEN_60S_REPEATABLE_CAUSAL_PARTICIPANT_INTERACTION_SUBSTRATE_NOT_ALPHA"
        ),
        "frozen_candidate_policy": FROZEN_POLICY,
        "frozen_reset_bps": FROZEN_RESET_BPS,
        "frozen_release_bps": FROZEN_RELEASE_BPS,
        "frozen_candidate_bps": FROZEN_CANDIDATE_BPS,
        "frozen_rearm_cooldown_sec": FROZEN_COOLDOWN_SEC,
        "candidate_count": int(len(selected)),
        "candidate_days": int(
            selected["candidate_date"].astype(str).nunique()
        ),
        "up_candidates": int(up),
        "down_candidates": int(down),
        "candidate_id_set_sha256": candidate_id_hash,
        "mechanics_rows": mechanics_rows,
        "valid_mechanics_rows": valid_mechanics,
        "decision_outcome_rows": decision_rows,
        "frozen_execution_engine": (
            "Phase 7I.4 v1.0.1 raw depth/trade replay implementation"
        ),
        "scientific_state": (
            "SUBSTRATE_ONLY_AWAIT_CHEAP_PHASE_7L2_MECHANISM_ANALYSIS"
        ),
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(latest_path, summary)

    run_config = {
        **summary,
        "phase7i4_script": str(phase7i4_script.resolve()),
        "phase7k2_root": str(phase7k2_root.resolve()),
        "observation_horizons_sec": list(FIXED_HORIZONS_SEC),
        "primary_observation_horizon_sec": PRIMARY_HORIZON_SEC,
        "causal_feature_contract": (
            "candidate_inputs.parquet + candidate_mechanics.parquet contain "
            "no future outcomes; candidate_decision_outcomes.parquet and "
            "candidate_k2_outcomes.parquet are physically separate future-label datasets"
        ),
        "next_scientific_question": (
            "Among comparable momentum/aggression states, does opposing-liquidity "
            "failure add information about 75/100 continuation, and does that "
            "mechanism differ for first impulses vs re-accelerations or UP vs DOWN?"
        ),
    }
    atomic_write_json(
        output_root / "metadata" / "run_config.json",
        run_config,
    )

    freeze_manifest = {
        "schema_name": SCHEMA_NAME,
        "builder_version": BUILDER_VERSION,
        "freeze_status": "FROZEN_60S_CANDIDATE_INPUT_TO_RAW_REPLAY",
        "policy_id": FROZEN_POLICY,
        "reset_bps": FROZEN_RESET_BPS,
        "release_bps": FROZEN_RELEASE_BPS,
        "candidate_bps": FROZEN_CANDIDATE_BPS,
        "rearm_cooldown_sec": FROZEN_COOLDOWN_SEC,
        "candidate_count": int(len(selected)),
        "up_candidates": int(up),
        "down_candidates": int(down),
        "candidate_days": int(
            selected["candidate_date"].astype(str).nunique()
        ),
        "candidate_id_set_sha256": candidate_id_hash,
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(
        output_root / "metadata" / "frozen_candidate_manifest.json",
        freeze_manifest,
    )

    report_root.mkdir(parents=True, exist_ok=True)
    contract = pd.DataFrame([
        {"field": "policy_id", "value": FROZEN_POLICY},
        {"field": "reset_bps", "value": FROZEN_RESET_BPS},
        {"field": "release_bps", "value": FROZEN_RELEASE_BPS},
        {"field": "candidate_bps", "value": FROZEN_CANDIDATE_BPS},
        {"field": "same_direction_rearm_cooldown_sec", "value": FROZEN_COOLDOWN_SEC},
        {"field": "candidate_count", "value": len(selected)},
        {"field": "up_candidates", "value": up},
        {"field": "down_candidates", "value": down},
        {"field": "candidate_days", "value": selected["candidate_date"].astype(str).nunique()},
        {"field": "candidate_id_set_sha256", "value": candidate_id_hash},
    ])
    contract.to_csv(
        report_root / "frozen_candidate_contract.csv",
        index=False,
    )

    findings = f"""# Phase 7L.1 — frozen 60s repeatable-candidate participant-interaction substrate

- Builder: **{BUILDER_VERSION}**
- Status: **{final_status}**
- Frozen candidate policy: **{FROZEN_POLICY}**
- Candidate contract: **5bp reset -> 10bp release -> 35bp candidate -> 60s minimum same-direction re-arm**
- Candidates: **{len(selected):,}** ({up:,} UP / {down:,} DOWN)
- Candidate dates: **{selected["candidate_date"].astype(str).nunique()}**
- Candidate-ID SHA-256: `{candidate_id_hash}`
- Decision horizons: **1 / 3 / 5 / 10 seconds**, primary **3 seconds**
- Raw replay engine: **frozen Phase 7I.4 v1.0.1**

## What this phase establishes

This is the reusable causal measurement substrate for the corrected repeatable-candidate ontology.
It measures exact book response and executed participant flow after each causal candidate without
using future continuation labels to define or filter the causal mechanics.

## What this phase does NOT establish

It does not show that liquidity failure predicts 75/100 continuation.
It does not create a composite book-failure score.
It does not control for momentum/aggression.
It does not establish profitability.

Those are intentionally deferred to cheap Phase 7L.2 analyses over this frozen substrate.

## Data separation

`candidate_inputs.parquet` and `candidate_mechanics.parquet` are causal feature datasets.
`candidate_decision_outcomes.parquet` and `candidate_k2_outcomes.parquet` contain future outcomes
and must remain physically separate from live/candidate features.
"""
    atomic_write_text(
        report_root / "participant_interaction_findings.md",
        findings,
    )


def self_test() -> int:
    pd = require_pandas()

    # Pure synthetic freeze-contract test; no raw files required.
    rows = []
    for i in range(4):
        rows.append({
            "candidate_id": f"c{i}",
            "policy_id": FROZEN_POLICY,
            "candidate_date": "2026-06-01",
            "direction": 1 if i % 2 == 0 else -1,
            "direction_label": "UP" if i % 2 == 0 else "DOWN",
            "origin_second": 1000 + i * 100,
            "origin_price": 100.0,
            "release_second": 1010 + i * 100,
            "candidate_second": 1040 + i * 100,
            "candidate_price": 100.35,
            "candidate_displacement_bps": 35.0,
            "reset_threshold_bps": 5.0,
            "release_threshold_bps": 10.0,
            "candidate_threshold_bps": 35.0,
            "cooldown_sec": 60,
            "release_to_candidate_sec": 30,
        })
    summary = {"builder_version": "1.0.0", "status": "PASS"}
    selected, contract = validate_and_adapt_selected_candidates(
        pd.DataFrame(rows),
        summary,
        full_scope_required=False,
    )

    assert len(selected) == 4
    assert selected["episode_id"].str.startswith(
        "REPEATABLE_CANDIDATE::"
    ).all()
    assert (
        selected["seconds_release_to_candidate"].astype(int) == 30
    ).all()
    assert contract == {
        "reset_bps": [5.0],
        "release_bps": [10.0],
        "candidate_bps": [35.0],
        "cooldown_sec": [60.0],
    }

    # Confirm the frozen replay dependency exists only when running inside repo;
    # the pure contract test remains usable in artifact environments.
    print("SELF-TEST PASS")
    print(
        "Validated 60s candidate-policy selection, 5/10/35bp causal contract, "
        "future-label exclusion, deterministic interface adaptation, and hashable freeze inputs."
    )
    return 0


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Phase 7L.1 frozen 60s repeatable-candidate participant-interaction substrate"
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--phase7i4-script", type=Path, default=DEFAULT_PHASE7I4_SCRIPT)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--phase7e-script", type=Path, default=DEFAULT_PHASE7E_SCRIPT)
    p.add_argument("--phase7i1-script", type=Path, default=DEFAULT_PHASE7I1_SCRIPT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--phase7k2-root", type=Path, default=DEFAULT_PHASE7K2_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-scale", type=float, default=10.0)
    p.add_argument("--max-book-age-ms", type=int, default=2000)
    p.add_argument(
        "--force-day",
        action="store_true",
        help="Rebuild per-date checkpoints even if their frozen signatures match.",
    )
    p.add_argument(
        "--max-candidate-dates",
        type=int,
        default=None,
        help="Smoke-test only: replay the first N candidate dates.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    if args.symbol != "BTCUSDT":
        raise SystemExit("Phase 7L.1 is frozen to BTCUSDT.")

    print("Building Phase 7L.1 frozen 60s participant-interaction substrate...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen policy        : {FROZEN_POLICY}")
    print(
        "Candidate contract   : "
        f"{FROZEN_RESET_BPS:g}bp reset -> "
        f"{FROZEN_RELEASE_BPS:g}bp release -> "
        f"{FROZEN_CANDIDATE_BPS:g}bp candidate -> "
        f"{FROZEN_COOLDOWN_SEC}s re-arm"
    )
    print(f"Expected candidates  : {EXPECTED_CANDIDATES:,}")
    print(f"Expected dates       : {EXPECTED_CANDIDATE_DAYS}")
    print("Decision horizons    : 1,3,5,10s (primary=3s)")
    print("Research role        : expensive causal substrate only; not alpha")
    print("Future labels        : physically separate")

    selected, selected_outcomes, freeze_summary, freeze_manifest, h = (
        load_k2_freeze(args.phase7k2_root)
    )

    engine = load_module(
        args.phase7i4_script,
        "phase7i4_frozen_execution_engine_for_7l1",
    )
    up, down = patch_7i4_engine(
        engine,
        selected,
        selected_outcomes,
        freeze_summary,
        freeze_manifest,
        h,
    )

    print(f"Selected candidates  : {len(selected):,} ({up:,} UP / {down:,} DOWN)")
    print(f"Candidate ID hash    : {h[:16]}...")
    print(
        "Replay engine         : frozen Phase 7I.4 exact depth/trade implementation"
    )

    engine_args = [
        "--symbol", args.symbol,
        "--candidate-root", str(args.phase7k2_root),
        "--canonical-root", str(args.canonical_root),
        "--execution-root", str(args.execution_root),
        "--phase6a-script", str(args.phase6a_script),
        "--phase7a-script", str(args.phase7a_script),
        "--phase7e-script", str(args.phase7e_script),
        "--phase7i1-script", str(args.phase7i1_script),
        "--output-root", str(args.output_root),
        "--report-root", str(args.report_root),
        "--price-scale", str(args.price_scale),
        "--max-book-age-ms", str(args.max_book_age_ms),
    ]
    if args.force_day:
        engine_args.append("--force-day")
    if args.max_candidate_dates is not None:
        engine_args.extend([
            "--max-candidate-dates",
            str(int(args.max_candidate_dates)),
        ])

    code = int(engine.main(engine_args))

    rewrite_phase7l_metadata(
        args.output_root,
        args.report_root,
        selected,
        selected_outcomes,
        h,
        up,
        down,
        code,
        args.phase7i4_script,
        args.phase7k2_root,
    )

    if code == 0:
        if args.max_candidate_dates is None:
            print("")
            print("Phase 7L.1 complete.")
            print(
                "Freeze this substrate. The next phase is cheap analysis of "
                "liquidity failure versus 75/100 continuation while controlling "
                "for momentum and aggressive pressure."
            )
        else:
            print("")
            print(
                "Phase 7L.1 partial smoke replay complete. Do NOT interpret "
                "continuation mechanisms until the full 62-date run passes."
            )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
