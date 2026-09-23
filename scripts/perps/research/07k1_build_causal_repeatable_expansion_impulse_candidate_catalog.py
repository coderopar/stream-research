#!/usr/bin/env python3
"""
Phase 7K.1 — Causal repeatable expansion-impulse candidate catalog
==================================================================

Scientific purpose
------------------
Replace the "one causal 35-bp candidate per broad directional episode" constraint
with a repeatable, price-only impulse-candidate ontology that can re-arm inside an
already-active physical leg.

This phase is intentionally CHEAP:
* Phase-6A one-second execution prices only for candidate generation.
* NO raw depth replay.
* NO raw trade replay.
* Phase-7I.9.2 alternate impulses are used only AFTER candidate generation for
  structural coverage auditing.
* Future 50/75/100 and path-quality outcomes are stored physically separately
  from causal candidate inputs.
* No ML and no profitability claim.

Core causal state
-----------------
For each direction independently:

1. Seek a local adverse extreme (low for UP, high for DOWN).
2. A 10-bp move away from that extreme causally confirms directional release.
3. First passage to 35 bps from the frozen origin creates a candidate.
4. After a candidate, that same direction is USED until a fresh reset occurs.
5. A same-direction reset may arm only after >=180 seconds from the prior
   same-direction candidate and after price pulls back from the post-candidate
   favorable extreme by the policy's reset threshold.
6. Track the deepest reset extreme.
7. Require a fresh 10-bp release away from that reset extreme.
8. First passage to 35 bps from the fresh origin creates another candidate.

The opposite direction runs symmetrically and independently. Therefore a large
counter-move can naturally create an opposite-direction candidate.

Predeclared re-arm policies
---------------------------
STRICT_RESET_10BP_180S
    Full 10-bp pullback before same-direction re-arm.

PRIMARY_RESET_5BP_180S
    Proposed primary ontology. A 5-bp pullback is enough to establish a fresh
    local reset, but a new 10-bp release is still required before the candidate
    can progress to 35 bps.

RELAXED_RESET_3BP_180S
    Sensitivity branch for shallow consolidations.

The 180-second minimum separation is inherited from the frozen Phase-7I.9.2
common-impulse audit. It is a structural anti-duplication interval, not an
outcome-optimized threshold.

Important:
This script DOES NOT automatically select/freeze a winning policy. The report is
an audit of candidate proliferation, alternate-impulse coverage, re-acceleration
coverage, base rates and spacing. We inspect those results before freezing an
ontology and before any expensive participant-interaction replay.

Frozen audit dependency
-----------------------
Phase-7I.9 builder metadata must report version 1.0.2.

Outputs
-------
data/derived/perps/research/causal_repeatable_expansion_impulse_candidates_v1/
    candidates.parquet
    candidate_outcomes.parquet
    candidate_to_impulse_mapping.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_causal_repeatable_expansion_impulse_candidates/
    impulse_candidate_policy_summary.csv
    impulse_candidate_base_rates.csv
    impulse_candidate_7i9_coverage.csv
    impulse_candidate_coverage_by_ordinal.csv
    impulse_candidate_mapping_audit.csv
    impulse_candidate_day_profile.csv
    impulse_candidate_spacing.csv
    impulse_candidate_invariants.csv
    impulse_candidate_findings.md

Interpretation discipline
-------------------------
* Candidate creation is causal and price-only.
* Phase-7I.9.2 labels never influence candidate creation.
* Outcome labels never influence reset/release thresholds.
* Candidate outputs contain no future outcome columns.
* Candidate-outcome and candidate-to-hindsight-impulse mappings are downstream
  audit products and must never be used as live causal features.
* A candidate that does not map to a Phase-7I.9.2 fixed-75 impulse is not
  automatically "false"; it simply did not become one of the independently
  defined 75-bp-within-15m audit impulses.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BUILDER_VERSION = "1.0.2"
SCHEMA_VERSION = 1
SCHEMA_NAME = "causal_repeatable_expansion_impulse_candidates_v1"
DEFAULT_SYMBOL = "BTCUSDT"

FROZEN_PHASE7A_VERSION = "1.0.0"
FROZEN_PHASE7I9_VERSION = "1.0.2"

RELEASE_BPS = 10.0
CANDIDATE_BPS = 35.0
MIN_REARM_SEPARATION_SEC = 180
MAX_OUTCOME_HORIZON_SEC = 1800
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
TARGETS_BPS = (50.0, 75.0, 100.0)
PATH_EFFICIENT_TARGET_BPS = 75.0
PATH_EFFICIENT_HORIZON_SEC = 600
PATH_EFFICIENT_MAX_POST_CANDIDATE_MAE_BPS = 15.0
IMPULSE_MAPPING_LOOKBACK_SEC = 900
DEFAULT_MAX_TRADE_STALE_SEC = 5

POLICIES: dict[str, float] = {
    "STRICT_RESET_10BP_180S": 10.0,
    "PRIMARY_RESET_5BP_180S": 5.0,
    "RELAXED_RESET_3BP_180S": 3.0,
}
PROPOSED_PRIMARY_POLICY = "PRIMARY_RESET_5BP_180S"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07a_build_large_move_event_catalog.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_PHASE7I1_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "causal_large_move_candidate_catalog_v1"
DEFAULT_PHASE7I9_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "causal_large_move_ontology_robustness_audit_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_causal_repeatable_expansion_impulse_candidates"


def require_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "Phase 7K.1 requires numpy and pandas. Activate the project .venv first."
        ) from exc
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        pa = None
        pq = None
    return np, pd, pa, pq


def require_parquet_stack():
    np, pd, pa, pq = require_stack()
    if pa is None or pq is None:
        raise SystemExit(
            "Phase 7K.1 Parquet I/O requires pyarrow. Activate the project .venv first."
        )
    return np, pd, pa, pq


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_time(sec: int | float | None) -> str | None:
    try:
        x = float(sec)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return datetime.fromtimestamp(int(x), tz=timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: Any, length: int = 24) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def directional_bps(origin_price: float, price: float, direction: int) -> float:
    return float(direction) * (float(price) / float(origin_price) - 1.0) * 10_000.0


def pullback_from_favorable_extreme_bps(extreme_price: float, price: float, direction: int) -> float:
    # UP: high -> lower price. DOWN: low -> higher price.
    return -float(direction) * (float(price) / float(extreme_price) - 1.0) * 10_000.0


def load_module(path: Path, name: str):
    import sys
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7k1-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7k1-", suffix=".csv", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7k1-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class DirectionState:
    direction: int
    phase: str = "SEEK_RELEASE"
    origin_price: float = math.nan
    origin_second: int = -1
    release_second: int = -1
    last_candidate_second: int = -1
    post_candidate_extreme_price: float = math.nan
    post_candidate_extreme_second: int = -1
    reset_origin_price: float = math.nan
    reset_origin_second: int = -1
    prior_candidate_second: int = -1
    prior_candidate_price: float = math.nan
    candidate_ordinal: int = 0
    origin_kind: str = "INITIAL_EXTREME"
    reset_pullback_bps: float = math.nan

    def reset_for_gap(self, sec: int, price: float) -> None:
        self.phase = "SEEK_RELEASE"
        self.origin_price = float(price)
        self.origin_second = int(sec)
        self.release_second = -1
        self.last_candidate_second = -1
        self.post_candidate_extreme_price = math.nan
        self.post_candidate_extreme_second = -1
        self.reset_origin_price = math.nan
        self.reset_origin_second = -1
        self.prior_candidate_second = -1
        self.prior_candidate_price = math.nan
        self.candidate_ordinal = 0
        self.origin_kind = "INITIAL_EXTREME"
        self.reset_pullback_bps = math.nan


def is_more_adverse(price: float, reference: float, direction: int) -> bool:
    if direction > 0:
        return float(price) < float(reference)
    return float(price) > float(reference)


def is_more_favorable(price: float, reference: float, direction: int) -> bool:
    if direction > 0:
        return float(price) > float(reference)
    return float(price) < float(reference)


def maybe_emit_candidate(
    state: DirectionState,
    sec: int,
    price: float,
    policy_id: str,
    reset_bps: float,
) -> dict[str, Any] | None:
    displacement = directional_bps(state.origin_price, price, state.direction)
    if displacement + 1e-12 < CANDIDATE_BPS:
        return None

    state.candidate_ordinal += 1
    candidate_id = stable_id(
        "7K1_CANDIDATE",
        policy_id,
        state.direction,
        state.origin_second,
        state.release_second,
        sec,
        state.candidate_ordinal,
    )
    previous_second = state.prior_candidate_second
    previous_price = state.prior_candidate_price
    row = {
        "candidate_id": candidate_id,
        "policy_id": policy_id,
        "reset_threshold_bps": float(reset_bps),
        "symbol": DEFAULT_SYMBOL,
        "direction": int(state.direction),
        "direction_label": "UP" if state.direction > 0 else "DOWN",
        "origin_kind": str(state.origin_kind),
        "origin_second": int(state.origin_second),
        "origin_time": iso_time(state.origin_second),
        "origin_price": float(state.origin_price),
        "release_second": int(state.release_second),
        "release_time": iso_time(state.release_second),
        "release_threshold_bps": float(RELEASE_BPS),
        "candidate_second": int(sec),
        "candidate_time": iso_time(sec),
        "candidate_date": datetime.fromtimestamp(int(sec), tz=timezone.utc).date().isoformat(),
        "candidate_price": float(price),
        "candidate_threshold_bps": float(CANDIDATE_BPS),
        "candidate_displacement_bps": float(displacement),
        "origin_to_release_sec": int(state.release_second - state.origin_second),
        "release_to_candidate_sec": int(sec - state.release_second),
        "origin_to_candidate_sec": int(sec - state.origin_second),
        "same_direction_candidate_ordinal": int(state.candidate_ordinal),
        "prior_same_direction_candidate_second": (
            int(previous_second) if previous_second >= 0 else math.nan
        ),
        "seconds_since_prior_same_direction_candidate": (
            int(sec - previous_second) if previous_second >= 0 else math.nan
        ),
        "prior_same_direction_candidate_price": (
            float(previous_price) if math.isfinite(previous_price) else math.nan
        ),
        "reset_pullback_bps": (
            float(state.reset_pullback_bps)
            if math.isfinite(state.reset_pullback_bps)
            else math.nan
        ),
        "candidate_contract": (
            f"PRICE_ONLY_RELEASE_{RELEASE_BPS:g}BP_"
            f"CANDIDATE_{CANDIDATE_BPS:g}BP_"
            f"REARM_{reset_bps:g}BP_MIN_{MIN_REARM_SEPARATION_SEC}S"
        ),
    }

    # The candidate has now consumed this same-direction impulse state.
    state.prior_candidate_second = int(sec)
    state.prior_candidate_price = float(price)
    state.last_candidate_second = int(sec)
    state.phase = "POST_CANDIDATE"
    state.post_candidate_extreme_price = float(price)
    state.post_candidate_extreme_second = int(sec)
    state.reset_origin_price = math.nan
    state.reset_origin_second = -1
    state.reset_pullback_bps = math.nan
    return row


def update_direction_state(
    state: DirectionState,
    sec: int,
    price: float,
    policy_id: str,
    reset_bps: float,
) -> dict[str, Any] | None:
    p = float(price)
    t = int(sec)

    if state.phase == "SEEK_RELEASE":
        if not math.isfinite(state.origin_price):
            state.origin_price = p
            state.origin_second = t
            return None
        if is_more_adverse(p, state.origin_price, state.direction):
            state.origin_price = p
            state.origin_second = t
            return None
        if directional_bps(state.origin_price, p, state.direction) + 1e-12 >= RELEASE_BPS:
            state.phase = "RELEASED"
            state.release_second = t
            # Large one-second jumps may cross release and candidate thresholds together.
            return maybe_emit_candidate(state, t, p, policy_id, reset_bps)
        return None

    if state.phase == "RELEASED":
        # Before 35 bps, a fresh adverse extreme invalidates the old origin and
        # restarts causal release discovery.
        if is_more_adverse(p, state.origin_price, state.direction):
            state.phase = "SEEK_RELEASE"
            state.origin_price = p
            state.origin_second = t
            state.release_second = -1
            state.origin_kind = "INITIAL_EXTREME" if state.candidate_ordinal == 0 else "REARM_RESET"
            return None
        return maybe_emit_candidate(state, t, p, policy_id, reset_bps)

    if state.phase == "POST_CANDIDATE":
        if is_more_favorable(p, state.post_candidate_extreme_price, state.direction):
            state.post_candidate_extreme_price = p
            state.post_candidate_extreme_second = t

        if t - state.last_candidate_second < MIN_REARM_SEPARATION_SEC:
            return None

        pullback = pullback_from_favorable_extreme_bps(
            state.post_candidate_extreme_price, p, state.direction
        )
        if pullback + 1e-12 >= reset_bps:
            state.phase = "RESETTING"
            state.reset_origin_price = p
            state.reset_origin_second = t
            state.reset_pullback_bps = float(pullback)
        return None

    if state.phase == "RESETTING":
        if is_more_adverse(p, state.reset_origin_price, state.direction):
            state.reset_origin_price = p
            state.reset_origin_second = t
            state.reset_pullback_bps = pullback_from_favorable_extreme_bps(
                state.post_candidate_extreme_price, p, state.direction
            )

        if directional_bps(state.reset_origin_price, p, state.direction) + 1e-12 >= RELEASE_BPS:
            state.phase = "RELEASED"
            state.origin_price = float(state.reset_origin_price)
            state.origin_second = int(state.reset_origin_second)
            state.release_second = t
            state.origin_kind = "REARM_RESET"
            return maybe_emit_candidate(state, t, p, policy_id, reset_bps)
        return None

    raise RuntimeError(f"Unknown direction-state phase: {state.phase}")


def build_candidates_for_policy(sec, price, policy_id: str, reset_bps: float, start_epoch: int, end_epoch: int):
    np, pd, _, _ = require_stack()
    rows: list[dict[str, Any]] = []
    up = DirectionState(+1)
    down = DirectionState(-1)
    prev_sec: int | None = None
    active = False

    for i in range(len(sec)):
        t = int(sec[i])
        p = float(price[i])
        valid = math.isfinite(p) and p > 0 and start_epoch <= t < end_epoch

        if not valid:
            active = False
            prev_sec = None
            continue

        if not active or prev_sec is None or t != prev_sec + 1:
            up.reset_for_gap(t, p)
            down.reset_for_gap(t, p)
            active = True
            prev_sec = t
            continue

        for state in (up, down):
            row = update_direction_state(state, t, p, policy_id, reset_bps)
            if row is not None:
                row["_price_index"] = int(i)
                rows.append(row)
        prev_sec = t

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["candidate_second", "policy_id", "direction"], kind="mergesort"
    ).reset_index(drop=True)


def first_passage_index(dr_bps, threshold: float) -> int | None:
    np, _, _, _ = require_stack()
    idx = np.flatnonzero(dr_bps + 1e-12 >= float(threshold))
    return int(idx[0]) if len(idx) else None


def build_candidate_outcomes(candidates, sec, price):
    np, pd, _, _ = require_stack()
    rows: list[dict[str, Any]] = []
    if candidates.empty:
        return pd.DataFrame()

    for r in candidates.to_dict("records"):
        i = int(r["_price_index"])
        csec = int(r["candidate_second"])
        origin = float(r["origin_price"])
        cprice = float(r["candidate_price"])
        direction = int(r["direction"])

        row: dict[str, Any] = {
            "candidate_id": str(r["candidate_id"]),
            "policy_id": str(r["policy_id"]),
            "direction": direction,
            "candidate_second": csec,
        }

        # Full 30-minute contiguous path if available; shorter outcomes remain
        # measurable even if the longest window is censored.
        max_j = min(len(sec) - 1, i + MAX_OUTCOME_HORIZON_SEC)
        # Trim at first gap.
        j = i
        while j < max_j and int(sec[j + 1]) == int(sec[j]) + 1 and math.isfinite(float(price[j + 1])):
            j += 1
        path = price[i : j + 1]
        path_sec = sec[i : j + 1]
        origin_dr = direction * (path / origin - 1.0) * 10_000.0
        candidate_dr = direction * (path / cprice - 1.0) * 10_000.0

        for target in TARGETS_BPS:
            hit = first_passage_index(origin_dr, target)
            hit_sec = int(path_sec[hit]) if hit is not None else None
            row[f"seconds_to_{int(target)}_from_candidate"] = (
                int(hit_sec - csec) if hit_sec is not None else math.nan
            )
            for h in OUTCOME_HORIZONS_SEC:
                row[f"reach_{int(target)}_within_{h // 60}m"] = bool(
                    hit_sec is not None and hit_sec - csec <= h
                )

        for h in OUTCOME_HORIZONS_SEC:
            avail = min(len(path), h + 1)
            if avail <= 0:
                row[f"mfe_{h // 60}m_bps"] = math.nan
                row[f"mae_{h // 60}m_bps"] = math.nan
                continue
            sub = candidate_dr[:avail]
            row[f"mfe_{h // 60}m_bps"] = float(max(0.0, float(np.nanmax(sub))))
            row[f"mae_{h // 60}m_bps"] = float(max(0.0, -float(np.nanmin(sub))))

        # Path-efficient continuation from the causal candidate timestamp:
        # reach the episode-origin 75-bp threshold within 10m while suffering
        # <=15bp adverse movement from the actual candidate price before target.
        hit75 = first_passage_index(origin_dr, PATH_EFFICIENT_TARGET_BPS)
        if hit75 is not None and int(path_sec[hit75]) - csec <= PATH_EFFICIENT_HORIZON_SEC:
            pre = candidate_dr[: hit75 + 1]
            pre_mae = float(max(0.0, -float(np.nanmin(pre))))
            row["pre75_mae_from_candidate_bps"] = pre_mae
            row["path_efficient_75_within_10m"] = bool(
                pre_mae <= PATH_EFFICIENT_MAX_POST_CANDIDATE_MAE_BPS + 1e-12
            )
        else:
            row["pre75_mae_from_candidate_bps"] = math.nan
            row["path_efficient_75_within_10m"] = False

        row["available_future_sec"] = int(path_sec[-1] - csec) if len(path_sec) else 0
        rows.append(row)

    return pd.DataFrame(rows)


def load_7i9(root: Path):
    _, _, _, pq = require_parquet_stack()
    meta_path = root / "metadata" / "latest_summary.json"
    impulses_path = root / "alternate_impulses.parquet"
    events_path = root / "alternate_events.parquet"
    missing = [str(p) for p in (meta_path, impulses_path, events_path) if not p.exists()]
    if missing:
        raise RuntimeError("Missing Phase-7I.9.2 audit outputs:\n  " + "\n  ".join(missing))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("builder_version")) != FROZEN_PHASE7I9_VERSION:
        raise RuntimeError(
            f"Phase 7K.1 requires Phase-7I.9 builder {FROZEN_PHASE7I9_VERSION}; "
            f"found {meta.get('builder_version')!r}"
        )
    impulses = pq.read_table(impulses_path).to_pandas()
    events = pq.read_table(events_path).to_pandas()
    return meta, impulses, events


def load_old_7i1_candidates(root: Path):
    _, pd, _, pq = require_parquet_stack()
    path = root / "candidates.parquet"
    if not path.exists():
        return pd.DataFrame()
    old = pq.read_table(path).to_pandas()
    # Frozen 7I.1 did not persist a release_second column.  Do NOT fabricate
    # one.  Its causal origin and 35-bp candidate timestamp are sufficient to
    # audit whether it was genuinely fresh after a previous hindsight impulse.
    required = {
        "candidate_id",
        "candidate_second",
        "direction",
        "origin_second",
    }
    missing = required - set(old.columns)
    if missing:
        raise RuntimeError(
            "Corrected 7K.1.2 mapping requires these frozen 7I.1 causal "
            f"candidate fields, but they are missing: {sorted(missing)}"
        )
    old = old.copy()
    old["policy_id"] = "OLD_7I1_FROZEN"
    old["candidate_id"] = old["candidate_id"].astype(str)
    # Explicitly mark the historical schema gap. Mapping logic branches on
    # availability rather than pretending release occurred at another time.
    if "release_second" not in old.columns:
        old["release_second"] = math.nan
    return old


def add_impulse_ordinals(impulses):
    _, pd, _, _ = require_stack()
    fixed = impulses[impulses["impulse_family"].astype(str) == "FIXED_75_PARENT"].copy()
    fixed = fixed.sort_values(
        ["phase7a_event_id","representative_hit_second","direction"], kind="mergesort"
    ).reset_index(drop=True)
    fixed["impulse_ordinal_in_physical_leg"] = (
        fixed.groupby("phase7a_event_id", dropna=False).cumcount() + 1
    )
    fixed["previous_impulse_hit_second"] = (
        fixed.groupby("phase7a_event_id", dropna=False)["representative_hit_second"].shift(1)
    )
    fixed["previous_impulse_hit_time"] = fixed["previous_impulse_hit_second"].map(iso_time)
    return fixed


def impulse_definition_flags(fixed_impulses, events):
    _, pd, _, _ = require_stack()
    defs = (
        events.groupby("impulse_id")["definition"]
        .apply(lambda x: set(map(str, x)))
        .to_dict()
    )
    out = fixed_impulses.copy()
    for definition in (
        "FIRST_PASSAGE_75_5M",
        "FIRST_PASSAGE_75_10M",
        "FIRST_PASSAGE_75_15M",
        "PATH_EFFICIENT_75_10M",
        "FIRST_PASSAGE_100_10M",
    ):
        out[definition] = out["impulse_id"].astype(str).map(
            lambda x, d=definition: d in defs.get(x, set())
        )
    return out


def map_candidates_to_impulses(candidates, impulses, policy_id: str):
    """
    Corrected greedy one-to-one hindsight association.

    Each fixed-75 impulse receives at most one candidate from this policy and
    each candidate can cover at most one impulse.

    FIRST impulse in a Phase-7A physical leg
    -----------------------------------------
    Choose the latest unused same-direction candidate satisfying:
        candidate origin <= release <= candidate <= impulse hit
        candidate is no more than 15 minutes before the impulse hit.

    LATER impulse / re-acceleration
    --------------------------------
    A candidate can receive credit ONLY if it represents a genuinely fresh
    causal structure after the preceding impulse:
        previous impulse hit
            < candidate origin
            <= candidate release
            <= candidate 35-bp time
            <= current impulse hit

    The same 15-minute maximum candidate-to-hit association window remains.

    This fixes v1.0.0's overly permissive rule, which could assign an old
    candidate from the previous expansion to a later impulse merely because it
    was unused, same-direction, and within 15 minutes of the later hit.
    """
    _, pd, _, _ = require_stack()
    c = candidates[candidates["policy_id"].astype(str) == policy_id].copy()
    if c.empty or impulses.empty:
        return pd.DataFrame()

    required_candidate_cols = {
        "candidate_id","candidate_second","direction","origin_second"
    }
    missing = required_candidate_cols - set(c.columns)
    if missing:
        raise RuntimeError(
            f"{policy_id} candidates missing corrected-mapping fields: {sorted(missing)}"
        )
    # New 7K.1 candidates have an explicit fresh-release timestamp. Frozen
    # 7I.1 does not. Keep that absence explicit and use the strongest causal
    # ordering supported by each frozen schema.
    if "release_second" not in c.columns:
        c["release_second"] = math.nan

    required_impulse_cols = {
        "impulse_id","direction","representative_hit_second",
        "impulse_ordinal_in_physical_leg","previous_impulse_hit_second",
        "phase7a_event_id",
    }
    missing_imp = required_impulse_cols - set(impulses.columns)
    if missing_imp:
        raise RuntimeError(
            f"Impulse audit missing corrected-mapping fields: {sorted(missing_imp)}"
        )

    rows = []
    used: set[str] = set()

    by_dir = {}
    for direction, g in c.groupby(pd.to_numeric(c["direction"], errors="coerce")):
        g = g.copy()
        for col in ("candidate_second","origin_second","release_second"):
            g[col] = pd.to_numeric(g[col], errors="coerce")
        g = g.sort_values("candidate_second", kind="mergesort").reset_index(drop=True)
        by_dir[int(direction)] = g

    impulses_sorted = impulses.sort_values(
        ["representative_hit_second","phase7a_event_id","impulse_ordinal_in_physical_leg"],
        kind="mergesort",
    )

    for imp in impulses_sorted.to_dict("records"):
        direction = int(imp["direction"])
        hit = int(imp["representative_hit_second"])
        ordinal = int(imp["impulse_ordinal_in_physical_leg"])
        prev_hit_raw = safe_float(imp.get("previous_impulse_hit_second"))
        prev_hit = int(prev_hit_raw) if math.isfinite(prev_hit_raw) else None

        g = by_dir.get(direction)
        chosen = None
        rejection_counts = {
            "too_old": 0,
            "used": 0,
            "bad_ordering": 0,
            "not_fresh_after_previous_impulse": 0,
        }

        if g is not None and len(g):
            secs = g["candidate_second"].astype(int).tolist()
            j = bisect_right(secs, hit) - 1

            while j >= 0:
                r = g.iloc[j]
                cid = str(r["candidate_id"])
                csec = int(r["candidate_second"])
                osec = int(r["origin_second"])
                rsec_raw = safe_float(r.get("release_second"))
                has_release = math.isfinite(rsec_raw)
                rsec = int(rsec_raw) if has_release else None

                if hit - csec > IMPULSE_MAPPING_LOOKBACK_SEC:
                    rejection_counts["too_old"] += 1
                    break

                if cid in used:
                    rejection_counts["used"] += 1
                    j -= 1
                    continue

                # Strongest ordering supported by the frozen candidate schema.
                # New 7K.1: origin <= release <= candidate <= hit.
                # Old 7I.1: origin <= candidate <= hit (release wasn't stored).
                causal_ordering_ok = (
                    osec <= csec <= hit
                    and (
                        (rsec is None)
                        or (osec <= rsec <= csec)
                    )
                )
                if not causal_ordering_ok:
                    rejection_counts["bad_ordering"] += 1
                    j -= 1
                    continue

                if ordinal > 1:
                    # The defining correction: later impulses require a fresh
                    # causal origin AFTER the preceding impulse hit. When an
                    # explicit release exists, it too must sit in causal order.
                    fresh_after_previous = (
                        prev_hit is not None
                        and prev_hit < osec <= csec <= hit
                        and (
                            (rsec is None)
                            or (osec <= rsec <= csec)
                        )
                    )
                    if not fresh_after_previous:
                        rejection_counts["not_fresh_after_previous_impulse"] += 1
                        j -= 1
                        continue

                chosen = r
                break

                j -= 1

        base = {
            "policy_id": policy_id,
            "impulse_id": str(imp["impulse_id"]),
            "phase7a_event_id": imp.get("phase7a_event_id"),
            "impulse_ordinal_in_physical_leg": ordinal,
            "previous_impulse_hit_second": (
                int(prev_hit) if prev_hit is not None else math.nan
            ),
            "previous_impulse_hit_time": iso_time(prev_hit),
            "impulse_hit_second": hit,
            "impulse_hit_time": iso_time(hit),
            "mapping_rule": (
                "FIRST_IMPULSE_LATEST_UNUSED_WITHIN_15M"
                if ordinal == 1
                else "REACCEL_FRESH_ORIGIN_AFTER_PREVIOUS_IMPULSE_HIT"
            ),
        }

        if chosen is not None:
            cid = str(chosen["candidate_id"])
            used.add(cid)
            rows.append({
                **base,
                "candidate_id": cid,
                "candidate_origin_second": int(chosen["origin_second"]),
                "candidate_origin_time": iso_time(int(chosen["origin_second"])),
                "candidate_release_second": (
                    int(chosen["release_second"])
                    if math.isfinite(safe_float(chosen.get("release_second")))
                    else math.nan
                ),
                "candidate_release_time": (
                    iso_time(int(chosen["release_second"]))
                    if math.isfinite(safe_float(chosen.get("release_second")))
                    else None
                ),
                "candidate_release_available": bool(
                    math.isfinite(safe_float(chosen.get("release_second")))
                ),
                "candidate_second": int(chosen["candidate_second"]),
                "candidate_time": iso_time(int(chosen["candidate_second"])),
                "candidate_lead_to_impulse_hit_sec": int(
                    hit - int(chosen["candidate_second"])
                ),
                "candidate_origin_after_previous_impulse_sec": (
                    int(chosen["origin_second"]) - prev_hit
                    if prev_hit is not None
                    else math.nan
                ),
                "candidate_origin_kind": chosen.get("origin_kind"),
                "mapped": True,
                "unmapped_reason": None,
                "rejected_not_fresh_count": int(
                    rejection_counts["not_fresh_after_previous_impulse"]
                ),
            })
        else:
            reason = (
                "NO_ELIGIBLE_FRESH_CANDIDATE_AFTER_PREVIOUS_IMPULSE"
                if ordinal > 1
                else "NO_ELIGIBLE_CANDIDATE_WITHIN_15M"
            )
            rows.append({
                **base,
                "candidate_id": None,
                "candidate_origin_second": math.nan,
                "candidate_origin_time": None,
                "candidate_release_second": math.nan,
                "candidate_release_time": None,
                "candidate_release_available": False,
                "candidate_second": math.nan,
                "candidate_time": None,
                "candidate_lead_to_impulse_hit_sec": math.nan,
                "candidate_origin_after_previous_impulse_sec": math.nan,
                "candidate_origin_kind": None,
                "mapped": False,
                "unmapped_reason": reason,
                "rejected_not_fresh_count": int(
                    rejection_counts["not_fresh_after_previous_impulse"]
                ),
            })

    return pd.DataFrame(rows)

def policy_summary(candidates):
    _, pd, _, _ = require_stack()
    rows = []
    for policy_id, g in candidates.groupby("policy_id", sort=False):
        secs = sorted(pd.to_numeric(g["candidate_second"], errors="coerce").dropna().astype(int))
        gaps = [b - a for a, b in zip(secs, secs[1:])]
        rows.append({
            "policy_id": policy_id,
            "reset_threshold_bps": safe_float(g["reset_threshold_bps"].iloc[0]) if "reset_threshold_bps" in g else math.nan,
            "candidates": int(len(g)),
            "candidate_dates": int(g["candidate_date"].nunique()) if "candidate_date" in g else math.nan,
            "up_candidates": int((pd.to_numeric(g["direction"], errors="coerce") > 0).sum()),
            "down_candidates": int((pd.to_numeric(g["direction"], errors="coerce") < 0).sum()),
            "rearm_candidates": int((g.get("origin_kind", "") == "REARM_RESET").sum()) if "origin_kind" in g else math.nan,
            "rearm_rate": float((g.get("origin_kind", "") == "REARM_RESET").mean()) if "origin_kind" in g and len(g) else math.nan,
            "median_candidates_per_day": float(g.groupby("candidate_date").size().median()) if "candidate_date" in g else math.nan,
            "max_candidates_in_day": int(g.groupby("candidate_date").size().max()) if "candidate_date" in g else math.nan,
            "median_global_gap_sec": float(pd.Series(gaps).median()) if gaps else math.nan,
        })
    return pd.DataFrame(rows)


def base_rates(candidates, outcomes):
    _, pd, _, _ = require_stack()
    d = candidates.merge(outcomes, on=["candidate_id","policy_id","direction","candidate_second"], how="inner")
    rows = []
    for policy_id, g in d.groupby("policy_id", sort=False):
        row = {"policy_id": policy_id, "candidates": int(len(g))}
        for target in TARGETS_BPS:
            for h in OUTCOME_HORIZONS_SEC:
                c = f"reach_{int(target)}_within_{h // 60}m"
                row[f"p{int(target)}_{h // 60}m"] = float(g[c].astype(bool).mean()) if len(g) else math.nan
        row["path_efficient_75_10m_rate"] = float(g["path_efficient_75_within_10m"].astype(bool).mean()) if len(g) else math.nan
        row["median_mfe_10m_bps"] = float(pd.to_numeric(g["mfe_10m_bps"], errors="coerce").median())
        row["median_mae_10m_bps"] = float(pd.to_numeric(g["mae_10m_bps"], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_report(impulses, mappings, policy_ids: Sequence[str]):
    _, pd, _, _ = require_stack()
    rows = []
    segments = [
        ("ALL_FIXED75_IMPULSES", pd.Series(True, index=impulses.index)),
        ("FIRST_IMPULSES", impulses["impulse_ordinal_in_physical_leg"] == 1),
        ("REACCEL_IMPULSES", impulses["impulse_ordinal_in_physical_leg"] > 1),
        ("75_WITHIN_5M", impulses["FIRST_PASSAGE_75_5M"].astype(bool)),
        ("75_WITHIN_10M", impulses["FIRST_PASSAGE_75_10M"].astype(bool)),
        ("PATH_EFFICIENT_75_10M", impulses["PATH_EFFICIENT_75_10M"].astype(bool)),
        ("100_WITHIN_10M", impulses["FIRST_PASSAGE_100_10M"].astype(bool)),
    ]
    for policy_id in policy_ids:
        m = mappings[mappings["policy_id"] == policy_id][["impulse_id","mapped"]].copy()
        x = impulses[["impulse_id"]].merge(m, on="impulse_id", how="left")
        mapped_series = x["mapped"].fillna(False).astype(bool)
        for segment, mask in segments:
            mask = mask.reset_index(drop=True)
            n = int(mask.sum())
            covered = int((mapped_series & mask).sum())
            rows.append({
                "policy_id": policy_id,
                "segment": segment,
                "impulses": n,
                "covered_impulses": covered,
                "coverage_rate": float(covered / n) if n else math.nan,
            })
    return pd.DataFrame(rows)


def coverage_by_ordinal(impulses, mappings, policy_ids: Sequence[str]):
    _, pd, _, _ = require_stack()
    rows = []
    for policy_id in policy_ids:
        m = mappings[mappings["policy_id"] == policy_id][["impulse_id","mapped"]]
        x = impulses.merge(m, on="impulse_id", how="left")
        x["mapped"] = x["mapped"].fillna(False).astype(bool)
        for ordinal, g in x.groupby("impulse_ordinal_in_physical_leg"):
            rows.append({
                "policy_id": policy_id,
                "impulse_ordinal_in_physical_leg": int(ordinal),
                "impulses": int(len(g)),
                "covered_impulses": int(g["mapped"].sum()),
                "coverage_rate": float(g["mapped"].mean()) if len(g) else math.nan,
            })
    return pd.DataFrame(rows)


def day_profile(candidates):
    _, pd, _, _ = require_stack()
    if candidates.empty:
        return pd.DataFrame()
    return (
        candidates.groupby(["policy_id","candidate_date","direction_label"])
        .size()
        .rename("candidates")
        .reset_index()
        .sort_values(["candidate_date","policy_id","direction_label"], kind="mergesort")
    )


def spacing_report(candidates):
    _, pd, _, _ = require_stack()
    rows = []
    for (policy_id, direction), g in candidates.groupby(["policy_id","direction"], sort=False):
        secs = sorted(pd.to_numeric(g["candidate_second"], errors="coerce").dropna().astype(int))
        gaps = [b - a for a, b in zip(secs, secs[1:])]
        if not gaps:
            continue
        q = pd.Series(gaps)
        rows.append({
            "policy_id": policy_id,
            "direction": int(direction),
            "direction_label": "UP" if int(direction) > 0 else "DOWN",
            "same_direction_gaps": int(len(gaps)),
            "p10_gap_sec": float(q.quantile(0.10)),
            "p25_gap_sec": float(q.quantile(0.25)),
            "median_gap_sec": float(q.median()),
            "p75_gap_sec": float(q.quantile(0.75)),
            "p90_gap_sec": float(q.quantile(0.90)),
            "share_gap_lt_5m": float((q < 300).mean()),
            "share_gap_lt_10m": float((q < 600).mean()),
        })
    return pd.DataFrame(rows)


def build_invariants(candidates, outcomes, impulses, mappings):
    _, pd, _, _ = require_stack()
    rows = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({
            "invariant": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    add("candidate_ids_unique", not candidates["candidate_id"].duplicated().any(), int(candidates["candidate_id"].duplicated().sum()), 0)
    add("candidate_outcomes_one_row_per_candidate", not outcomes["candidate_id"].duplicated().any(), int(outcomes["candidate_id"].duplicated().sum()), 0)
    add("candidate_outcomes_cover_all_candidates", set(candidates["candidate_id"]) == set(outcomes["candidate_id"]), len(outcomes), len(candidates))
    add("candidate_origin_before_or_equal_release",
        bool((pd.to_numeric(candidates["origin_second"]) <= pd.to_numeric(candidates["release_second"])).all()),
        "ordering", True)
    add("candidate_release_before_or_equal_candidate",
        bool((pd.to_numeric(candidates["release_second"]) <= pd.to_numeric(candidates["candidate_second"])).all()),
        "ordering", True)
    add("candidate_displacement_at_least_35bp",
        bool((pd.to_numeric(candidates["candidate_displacement_bps"], errors="coerce") + 1e-9 >= CANDIDATE_BPS).all()),
        float(pd.to_numeric(candidates["candidate_displacement_bps"], errors="coerce").min()),
        f">={CANDIDATE_BPS}")
    leakage_tokens = ("reach_", "mfe_", "mae_", "future", "outcome", "path_efficient", "seconds_to_50", "seconds_to_75", "seconds_to_100")
    leakage_cols = [
        c for c in candidates.columns
        if any(tok in c.lower() for tok in leakage_tokens)
    ]
    add("candidate_table_has_no_future_label_names", len(leakage_cols) == 0, leakage_cols, [])
    add("fixed75_impulses_nonempty", len(impulses) > 0, len(impulses), ">0")
    if len(mappings):
        mapped_only = mappings[mappings["mapped"].fillna(False)].copy()
        dup = mapped_only.duplicated(["policy_id","candidate_id"]).sum()
        add("mapped_candidate_used_at_most_once_per_policy", int(dup) == 0, int(dup), 0)

        reacc = mapped_only[
            pd.to_numeric(mapped_only["impulse_ordinal_in_physical_leg"], errors="coerce") > 1
        ].copy()
        if len(reacc):
            prev = pd.to_numeric(reacc["previous_impulse_hit_second"], errors="coerce")
            origin = pd.to_numeric(reacc["candidate_origin_second"], errors="coerce")
            release = pd.to_numeric(reacc["candidate_release_second"], errors="coerce")
            cand = pd.to_numeric(reacc["candidate_second"], errors="coerce")
            hit = pd.to_numeric(reacc["impulse_hit_second"], errors="coerce")
            release_available = reacc["candidate_release_available"].fillna(False).astype(bool)

            base_fresh = (prev < origin) & (origin <= cand) & (cand <= hit)
            release_order_ok = (
                ~release_available
                | ((origin <= release) & (release <= cand))
            )
            fresh_ok = base_fresh & release_order_ok
            add(
                "mapped_reacceleration_candidates_are_fresh_after_previous_impulse",
                bool(fresh_ok.all()),
                int((~fresh_ok).sum()),
                0,
            )

            new_rows = reacc["policy_id"].astype(str) != "OLD_7I1_FROZEN"
            new_missing_release = int(
                (new_rows & ~release_available).sum()
            )
            add(
                "new_7k1_mapped_reaccelerations_have_release_timestamp",
                new_missing_release == 0,
                new_missing_release,
                0,
            )
        else:
            add(
                "mapped_reacceleration_candidates_are_fresh_after_previous_impulse",
                True,
                "no mapped reaccelerations",
                0,
            )
    else:
        add("mapped_candidate_used_at_most_once_per_policy", False, "no mappings", "nonempty")
        add(
            "mapped_reacceleration_candidates_are_fresh_after_previous_impulse",
            False,
            "no mappings",
            "nonempty",
        )
    # Rearm candidates must respect minimum same-direction separation.
    rearm = candidates[candidates["origin_kind"] == "REARM_RESET"]
    sep = pd.to_numeric(rearm["seconds_since_prior_same_direction_candidate"], errors="coerce")
    add("rearm_candidates_respect_180s_minimum",
        bool((sep >= MIN_REARM_SEPARATION_SEC).all()) if len(rearm) else True,
        float(sep.min()) if len(rearm) else "no rearm candidates",
        f">={MIN_REARM_SEPARATION_SEC}")
    return pd.DataFrame(rows)


def findings_text(summary: Mapping[str, Any], policy_summary_df, base_rates_df, coverage_df) -> str:
    lines = [
        "# Phase 7K.1 — Causal repeatable expansion-impulse candidate catalog",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{summary['status']}**",
        f"- Candidate policies: **{', '.join(POLICIES)}**",
        f"- Proposed primary (NOT YET FROZEN): **{PROPOSED_PRIMARY_POLICY}**",
        f"- Frozen Phase-7I.9 audit version: **{FROZEN_PHASE7I9_VERSION}**",
        "",
        "## Structural policy summary",
        "",
    ]
    for _, r in policy_summary_df.iterrows():
        lines.append(
            f"- {r['policy_id']}: candidates={int(r['candidates']):,}; "
            f"rearm={int(r['rearm_candidates']):,} "
            f"({100*safe_float(r['rearm_rate']):.1f}%); "
            f"median/day={safe_float(r['median_candidates_per_day']):.1f}; "
            f"max/day={int(r['max_candidates_in_day'])}."
        )
    lines += ["", "## Outcome base rates (audit only)", ""]
    for _, r in base_rates_df.iterrows():
        lines.append(
            f"- {r['policy_id']}: P75<=5m={100*safe_float(r['p75_5m']):.1f}%; "
            f"P75<=10m={100*safe_float(r['p75_10m']):.1f}%; "
            f"P75<=30m={100*safe_float(r['p75_30m']):.1f}%; "
            f"path-efficient75/10={100*safe_float(r['path_efficient_75_10m_rate']):.1f}%."
        )
    lines += ["", "## Frozen 7I.9.2 impulse coverage", ""]
    for policy in [*POLICIES.keys(), "OLD_7I1_FROZEN"]:
        x = coverage_df[(coverage_df["policy_id"] == policy) & (coverage_df["segment"].isin(["ALL_FIXED75_IMPULSES","REACCEL_IMPULSES"]))]
        if len(x):
            vals = {r["segment"]: r for _, r in x.iterrows()}
            allr = vals.get("ALL_FIXED75_IMPULSES")
            rr = vals.get("REACCEL_IMPULSES")
            if allr is not None and rr is not None:
                lines.append(
                    f"- {policy}: all={100*safe_float(allr['coverage_rate']):.1f}% "
                    f"({int(allr['covered_impulses'])}/{int(allr['impulses'])}); "
                    f"re-accelerations={100*safe_float(rr['coverage_rate']):.1f}% "
                    f"({int(rr['covered_impulses'])}/{int(rr['impulses'])})."
                )
    lines += [
        "",
        "## Interpretation discipline",
        "",
        "- This builder does not select a winning re-arm policy.",
        "- Prefer structural coverage/duplication/base-rate trade-offs over chasing the highest future continuation rate.",
        "- Phase-7I.9.2 labels are hindsight audit targets only.",
        "- Re-acceleration coverage is credited only when candidate origin/reset occurs AFTER the previous impulse hit.",
        "- Do not replay raw depth until one candidate ontology is frozen.",
        "- If the relaxed branch creates many extra candidates without materially improving re-acceleration coverage, reject it.",
        "- If the strict branch preserves precision but still misses most later impulses, it is too close to the old one-candidate-per-episode behavior.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    np, pd, _, _ = require_stack()

    # Synthetic one-second path:
    # 1) clean UP candidate from 100.00 -> 100.35;
    # 2) wait >180s while extending slightly;
    # 3) pull back ~6bp (enough for 3/5bp policies, not 10bp);
    # 4) fresh +35bp from reset low => second UP candidate only in 3/5 branches.
    n = 900
    sec = np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64)
    price = np.full(n, 100.0, dtype=float)
    price[:50] = 100.0
    price[50:151] = np.linspace(100.0, 100.40, 101)
    price[151:400] = np.linspace(100.40, 100.44, 249)
    price[400:451] = np.linspace(100.44, 100.379, 51)  # ~6.1 bp reset
    price[451:651] = np.linspace(100.379, 100.80, 200)
    price[651:] = 100.80

    start_epoch = int(sec[0])
    end_epoch = int(sec[-1] + 1)

    built = {}
    for policy, reset in POLICIES.items():
        c = build_candidates_for_policy(sec, price, policy, reset, start_epoch, end_epoch)
        built[policy] = c

    assert len(built["STRICT_RESET_10BP_180S"][built["STRICT_RESET_10BP_180S"]["direction"] == 1]) == 1
    assert len(built["PRIMARY_RESET_5BP_180S"][built["PRIMARY_RESET_5BP_180S"]["direction"] == 1]) >= 2
    assert len(built["RELAXED_RESET_3BP_180S"][built["RELAXED_RESET_3BP_180S"]["direction"] == 1]) >= 2

    p = built["PRIMARY_RESET_5BP_180S"]
    re = p[(p["direction"] == 1) & (p["origin_kind"] == "REARM_RESET")]
    assert len(re) >= 1
    assert float(re["seconds_since_prior_same_direction_candidate"].min()) >= MIN_REARM_SEPARATION_SEC
    assert float(re["reset_pullback_bps"].min()) >= 5.0 - 1e-9

    # Corrected mapping regression:
    # An old candidate before impulse 1 must NOT receive credit for impulse 2.
    # A fresh candidate whose origin is after impulse 1 hit SHOULD receive credit.
    test_candidates = pd.DataFrame([
        {
            "candidate_id": "old_a",
            "policy_id": "TEST",
            "direction": 1,
            "origin_second": 100,
            "release_second": 110,
            "candidate_second": 130,
            "origin_kind": "INITIAL_EXTREME",
        },
        {
            "candidate_id": "fresh_b",
            "policy_id": "TEST",
            "direction": 1,
            "origin_second": 410,
            "release_second": 430,
            "candidate_second": 460,
            "origin_kind": "REARM_RESET",
        },
    ])
    test_impulses = pd.DataFrame([
        {
            "impulse_id": "imp1",
            "phase7a_event_id": "leg1",
            "direction": 1,
            "representative_hit_second": 400,
            "impulse_ordinal_in_physical_leg": 1,
            "previous_impulse_hit_second": math.nan,
        },
        {
            "impulse_id": "imp2",
            "phase7a_event_id": "leg1",
            "direction": 1,
            "representative_hit_second": 600,
            "impulse_ordinal_in_physical_leg": 2,
            "previous_impulse_hit_second": 400,
        },
    ])
    mapped_test = map_candidates_to_impulses(test_candidates, test_impulses, "TEST")
    m1 = mapped_test[mapped_test["impulse_id"] == "imp1"].iloc[0]
    m2 = mapped_test[mapped_test["impulse_id"] == "imp2"].iloc[0]
    assert bool(m1["mapped"]) and m1["candidate_id"] == "old_a"
    assert bool(m2["mapped"]) and m2["candidate_id"] == "fresh_b"
    assert int(m2["candidate_origin_second"]) > int(m2["previous_impulse_hit_second"])

    stale_only = test_candidates[test_candidates["candidate_id"] == "old_a"].copy()
    mapped_stale = map_candidates_to_impulses(stale_only, test_impulses, "TEST")
    second_stale = mapped_stale[mapped_stale["impulse_id"] == "imp2"].iloc[0]
    assert not bool(second_stale["mapped"])

    # Frozen-7I.1 schema regression: no release_second exists. A candidate with
    # an origin after the previous impulse may still be fairly credited using
    # origin <= candidate <= hit, while a stale origin must not be credited.
    old_schema = pd.DataFrame([
        {
            "candidate_id": "old_schema_fresh",
            "policy_id": "OLDTEST",
            "direction": 1,
            "origin_second": 420,
            "candidate_second": 470,
            "origin_kind": "INITIAL_EXTREME",
        }
    ])
    mapped_old_schema = map_candidates_to_impulses(old_schema, test_impulses, "OLDTEST")
    old_second = mapped_old_schema[mapped_old_schema["impulse_id"] == "imp2"].iloc[0]
    assert bool(old_second["mapped"])
    assert not bool(old_second["candidate_release_available"])

    # Candidate table must remain causal-label clean.
    leakage_tokens = ("reach_", "mfe_", "mae_", "future", "outcome", "path_efficient")
    assert not [
        c for c in p.columns
        if any(tok in c.lower() for tok in leakage_tokens)
    ]

    print("SELF-TEST PASS")
    print("Validated causal release->35bp candidate creation, 180s re-arm separation, repeatable same-direction impulses, schema-aware fresh-origin re-acceleration mapping for new 7K.1 and frozen 7I.1 candidates, and future-label separation.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7K.1 causal repeatable expansion-impulse candidate catalog")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--phase7i1-root", type=Path, default=DEFAULT_PHASE7I1_ROOT)
    p.add_argument("--phase7i9-root", type=Path, default=DEFAULT_PHASE7I9_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--max-trade-stale-sec", type=int, default=DEFAULT_MAX_TRADE_STALE_SEC)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    np, pd, _, _ = require_stack()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise SystemExit("--start-date must be <= --end-date")
    if args.symbol != DEFAULT_SYMBOL:
        raise SystemExit(f"Phase 7K.1 is frozen to {DEFAULT_SYMBOL}; received {args.symbol}")
    if args.max_trade_stale_sec != DEFAULT_MAX_TRADE_STALE_SEC:
        raise SystemExit(
            f"Phase 7K.1 is frozen to max_trade_stale_sec={DEFAULT_MAX_TRADE_STALE_SEC}; "
            f"received {args.max_trade_stale_sec}"
        )

    phase7a = load_module(args.phase7a_script, "phase7a_frozen_for_7k1")
    if getattr(phase7a, "BUILDER_VERSION", None) != FROZEN_PHASE7A_VERSION:
        raise RuntimeError(
            f"Phase 7K.1 requires Phase-7A {FROZEN_PHASE7A_VERSION}; "
            f"found {getattr(phase7a, 'BUILDER_VERSION', None)!r}"
        )

    print("Building Phase 7K.1 causal repeatable expansion-impulse candidates...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Frozen Phase-7A       : {phase7a.BUILDER_VERSION}")
    print(f"Required Phase-7I.9   : {FROZEN_PHASE7I9_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Dates                 : {start} to {end}")
    print(f"Release               : {RELEASE_BPS:g} bp")
    print(f"Candidate              : {CANDIDATE_BPS:g} bp")
    print(f"Minimum re-arm gap    : {MIN_REARM_SEPARATION_SEC}s")
    print(f"Policies               : {', '.join(POLICIES)}")
    print("Raw depth replay      : NONE")
    print("Raw trade replay      : NONE")
    print("Research role         : causal candidate-ontology audit; not alpha")

    print("[1/8] Loading Phase-6A one-second execution-price range...")
    days = []
    d = start - timedelta(days=1)
    while d <= end + timedelta(days=1):
        pth = phase7a.execution_day_path(args.execution_root, args.symbol, d)
        if pth.exists():
            days.append(d)
        d += timedelta(days=1)
    if not days:
        raise RuntimeError("No Phase-6A execution partitions found")

    frame = phase7a.load_execution_range(args.execution_root, args.symbol, days)
    sec, price, price_source = phase7a.build_catalog_price(
        frame, args.max_trade_stale_sec, "last"
    )
    print(f"      rows={len(sec):,} price_source={price_source}")

    start_epoch = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_epoch = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp()
    )

    print("[2/8] Generating causal candidates under predeclared re-arm policies...")
    policy_frames = []
    for policy_id, reset_bps in POLICIES.items():
        c = build_candidates_for_policy(
            sec, price, policy_id, reset_bps, start_epoch, end_epoch
        )
        print(
            f"      {policy_id}: candidates={len(c):,} "
            f"rearm={(c['origin_kind'] == 'REARM_RESET').sum() if len(c) else 0:,}"
        )
        policy_frames.append(c)
    candidates_runtime = pd.concat(policy_frames, ignore_index=True)
    candidates_runtime = candidates_runtime.sort_values(
        ["policy_id","candidate_second","direction"], kind="mergesort"
    ).reset_index(drop=True)

    print("[3/8] Building physically separate future outcome labels...")
    outcomes = build_candidate_outcomes(candidates_runtime, sec, price)
    print(f"      outcome rows={len(outcomes):,}")

    print("[4/8] Loading corrected Phase-7I.9.2 common impulse audit...")
    audit_meta, impulses_raw, events = load_7i9(args.phase7i9_root)
    impulses = impulse_definition_flags(add_impulse_ordinals(impulses_raw), events)
    print(
        f"      fixed75 impulses={len(impulses):,} "
        f"reaccelerations={(impulses['impulse_ordinal_in_physical_leg'] > 1).sum():,}"
    )

    print("[5/8] Mapping candidate policies to frozen alternate impulses...")
    mappings = []
    policy_ids = list(POLICIES)
    for policy_id in policy_ids:
        m = map_candidates_to_impulses(candidates_runtime, impulses, policy_id)
        mappings.append(m)
        covered = int(m["mapped"].sum()) if len(m) else 0
        print(f"      {policy_id}: covered={covered:,}/{len(impulses):,}")

    old = load_old_7i1_candidates(args.phase7i1_root)
    if len(old):
        # Old table may not contain fields used only by new mapping output.
        m_old = map_candidates_to_impulses(old, impulses, "OLD_7I1_FROZEN")
        mappings.append(m_old)
        policy_ids.append("OLD_7I1_FROZEN")
        print(f"      OLD_7I1_FROZEN: covered={int(m_old['mapped'].sum()):,}/{len(impulses):,}")
    mapping = pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()

    print("[6/8] Building structural/base-rate/coverage reports...")
    public_candidates = candidates_runtime.drop(columns=["_price_index"]).copy()
    pol_summary = policy_summary(public_candidates)
    rates = base_rates(public_candidates, outcomes)
    coverage = coverage_report(impulses, mapping, policy_ids)
    ordinal = coverage_by_ordinal(impulses, mapping, policy_ids)
    days_df = day_profile(public_candidates)
    spacing = spacing_report(public_candidates)

    print("[7/8] Running invariants...")
    invariants = build_invariants(
        public_candidates, outcomes, impulses, mapping
    )
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    candidate_hashes = {
        policy: hashlib.sha256(
            "\n".join(
                sorted(
                    public_candidates.loc[
                        public_candidates["policy_id"] == policy, "candidate_id"
                    ].astype(str)
                )
            ).encode("utf-8")
        ).hexdigest()
        for policy in POLICIES
    }
    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "status": status,
        "built_at_utc": utc_now_iso(),
        "symbol": args.symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "price_source": price_source,
        "candidate_count_total_across_policies": int(len(public_candidates)),
        "candidate_count_by_policy": {
            p: int((public_candidates["policy_id"] == p).sum()) for p in POLICIES
        },
        "candidate_id_set_sha256_by_policy": candidate_hashes,
        "fixed75_impulse_count": int(len(impulses)),
        "reacceleration_impulse_count": int(
            (impulses["impulse_ordinal_in_physical_leg"] > 1).sum()
        ),
        "proposed_primary_policy": PROPOSED_PRIMARY_POLICY,
        "scientific_state": "AUDIT_READY_NOT_YET_FROZEN",
        "failed_invariants": int(len(failed)),
    }
    run_config = {
        **summary,
        "release_bps": RELEASE_BPS,
        "candidate_bps": CANDIDATE_BPS,
        "minimum_rearm_separation_sec": MIN_REARM_SEPARATION_SEC,
        "policies": POLICIES,
        "outcome_horizons_sec": list(OUTCOME_HORIZONS_SEC),
        "targets_bps": list(TARGETS_BPS),
        "path_efficient_target_bps": PATH_EFFICIENT_TARGET_BPS,
        "path_efficient_horizon_sec": PATH_EFFICIENT_HORIZON_SEC,
        "path_efficient_max_post_candidate_mae_bps": PATH_EFFICIENT_MAX_POST_CANDIDATE_MAE_BPS,
        "impulse_mapping_lookback_sec": IMPULSE_MAPPING_LOOKBACK_SEC,
        "reacceleration_mapping_rule": (
            "all policies require previous_impulse_hit < candidate_origin "
            "<= candidate_second <= current_impulse_hit; new 7K.1 policies "
            "additionally require candidate_origin <= candidate_release "
            "<= candidate_second; frozen 7I.1 has no stored release_second"
        ),
        "frozen_phase7i9_builder_version": audit_meta.get("builder_version"),
        "execution_root": str(args.execution_root),
        "phase7i1_root": str(args.phase7i1_root),
        "phase7i9_root": str(args.phase7i9_root),
    }
    findings = findings_text(summary, pol_summary, rates, coverage)

    print("[8/8] Writing atomic datasets and reports...")
    atomic_write_parquet(public_candidates, args.output_root / "candidates.parquet")
    atomic_write_parquet(outcomes, args.output_root / "candidate_outcomes.parquet")
    atomic_write_parquet(mapping, args.output_root / "candidate_to_impulse_mapping.parquet")
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", run_config)

    atomic_write_csv(pol_summary, args.report_root / "impulse_candidate_policy_summary.csv")
    atomic_write_csv(rates, args.report_root / "impulse_candidate_base_rates.csv")
    atomic_write_csv(coverage, args.report_root / "impulse_candidate_7i9_coverage.csv")
    atomic_write_csv(ordinal, args.report_root / "impulse_candidate_coverage_by_ordinal.csv")
    atomic_write_csv(mapping, args.report_root / "impulse_candidate_mapping_audit.csv")
    atomic_write_csv(days_df, args.report_root / "impulse_candidate_day_profile.csv")
    atomic_write_csv(spacing, args.report_root / "impulse_candidate_spacing.csv")
    atomic_write_csv(invariants, args.report_root / "impulse_candidate_invariants.csv")
    atomic_write_text(args.report_root / "impulse_candidate_findings.md", findings)

    print("")
    print(f"Phase 7K.1 status: {status}")
    for policy in POLICIES:
        n = int((public_candidates["policy_id"] == policy).sum())
        print(f"{policy:28s}: {n:,} candidates")
    print(f"Fixed75 audit impulses : {len(impulses):,}")
    print(f"Failed invariants      : {len(failed)}")
    print(f"Scientific state       : AUDIT_READY_NOT_YET_FROZEN")
    print(f"Reports                : {args.report_root}")
    if len(failed):
        print("")
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
