#!/usr/bin/env python3
"""Phase 7E: exact raw-depth continuation mechanics for matched physical moves.

Research role
-------------
Phase 7D.1 established that ordinary <35-bp timestamps generally lack common
support with large-move physical extrema, while genuine 35_ONLY physical events
provide credible matched controls for the 50+/75+ continuation question.

Phase 7E therefore asks a narrower mechanism question:

    Starting from matched physical releases, why does one move terminate near
    35 bps while another continues to 50/75+ bps?

The primary population is the frozen Phase-7D.1 analysis:

    75_PLUS | THIRTYFIVE_ONLY | REGIME_MATCHED | PRIMARY

Robustness analyses use the frozen 75+ relaxed regime sample and the larger 50+
regime/strict-state samples.  100+ remains exploratory because Phase 7D.1 did
not produce an adequately sized balanced 100+ common-support sample.

What is new in 7E
-----------------
This script returns to canonical Binance USD-M depth diffs and reconstructs the
book with the audited Phase-6A contract:

    complete local checkpoint bootstrap
      -> exact pu == snapshot.last_u handshake
      -> authoritative continuous depth diffs
      -> later snapshots audit-only
      -> invalidate on genuine pu gap / crossed book

At each hindsight event stage, the exact price levels that are visible within
5 bps of the stage-end mid are FROZEN.  The script then follows those SAME PRICE
LEVELS forward rather than repeatedly asking what lies inside a moving 5-bp
band.  This distinction is central: it separates exact same-price queue behavior
from price translation through a moving measurement band.

For support-side and opposing-side frozen levels it measures, among other things:

* exact gross depletion of the stage-start levels,
* exact same-price positive quantity updates,
* same-price refill after an observed depletion,
* first-refill latency,
* durable refill at the end of the measurement window,
* retained initial queue share and complete level disappearance,
* new near-touch level additions at prices absent from the frozen cohort,
* additions 5-10 bps farther from the stage-start mid (an OUTWARD REPLACEMENT
  PROXY, never identified as the same order or participant),
* raw forward aggressive trade pressure and direction-normalized mid movement.

Fixed follow-up windows are 1/3/5/10 seconds.  Stage-to-stage transition windows
are also measured for:

    VOL_SCALED_DEPARTURE -> FIXED_10BP_DEPARTURE
    FIXED_10BP_DEPARTURE -> PRIOR_60S_RANGE_BREAK
    PRIOR_60S_RANGE_BREAK -> HIT_35BP
    FIXED_10BP_DEPARTURE -> HIT_35BP

Important interpretation limits
-------------------------------
* Event directions, extrema, stages, 35_ONLY labels and 50+/75+ continuation
  labels are hindsight-defined.  This remains retrospective mechanism research.
* A depth decrease is an exact observed book-quantity decrease, but the Binance
  diff stream alone does NOT identify whether the missing quantity traded,
  cancelled, or was modified for another reason.  Aggregate raw aggressive
  trades are reported alongside depth changes; no per-order cancellation label
  is invented.
* "Outward replacement" is a price-level addition proxy at farther prices.  It
  does NOT prove that the same market maker cancelled and reposted an order.
* Same-price refill is exact at the price level, but participant/order identity
  is unavailable.
* No prospective candidate scan, classifier, p-value claim, or tradable-edge
  claim is made here.

The script never modifies Phase-6A/7A/7C/7D.1 audited upstream datasets.  Raw
replay outputs are checkpointed by event onset date, written atomically, and
reused only when the selected-event/input signature matches.  Partial
``--max-onset-dates`` runs are smoke/replay checks: they do not require complete
matched parent/control closure and must not be interpreted as mechanism reports.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import gc
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILDER_VERSION = "1.0.3"
SCHEMA_VERSION = 1
SCHEMA_NAME = "large_move_raw_depth_continuation_mechanics_v1"
DEFAULT_SYMBOL = "BTCUSDT"
PRICE_SCALE = 10.0
MAX_BOOK_AGE_MS = 2_000
FIXED_HORIZONS_SEC = (1, 3, 5, 10)
FROZEN_NEAR_BPS = 5.0
OUTER_BAND_BPS = 10.0
MIN_INTERPRETABLE_PARENTS = 30
BALANCE_GOOD_MAX_ABS_SMD = 0.20
MIN_RESPONSE_NET_FLOW_QUOTE = 50_000.0
MIN_VALID_CONTROLS_PER_PARENT = 2

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_PHASE7C_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07c_build_large_move_participant_behavior_atlas.py"
DEFAULT_PHASE7D1_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07d1_build_common_support_matched_controls.py"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_PHASE7D1_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_common_support_matched_controls_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_large_move_raw_depth_mechanics"

# Frozen analysis choices.  These are chosen from the Phase-7D.1 balance audit,
# not tuned using any Phase-7E depth outcome.
ANALYSIS_SPECS = (
    ("PRIMARY", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
    ("ROBUST_75_RELAXED", "75_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|RELAXED"),
    ("ROBUST_50_STRICT", "50_PLUS|THIRTYFIVE_ONLY|STRICT_STATE_MATCHED|RELAXED"),
    ("ROBUST_50_REGIME", "50_PLUS|THIRTYFIVE_ONLY|REGIME_MATCHED|PRIMARY"),
)
PRIMARY_ANALYSIS_ID = ANALYSIS_SPECS[0][1]

STAGES = (
    "EXTREMUM_HINDSIGHT",
    "VOL_SCALED_DEPARTURE",
    "FIXED_10BP_DEPARTURE",
    "PRIOR_60S_RANGE_BREAK",
    "HIT_35BP",
)
TRANSITIONS = (
    ("VOL_SCALED_DEPARTURE", "FIXED_10BP_DEPARTURE", "TO_10BP"),
    ("FIXED_10BP_DEPARTURE", "PRIOR_60S_RANGE_BREAK", "TO_RANGE_BREAK"),
    ("PRIOR_60S_RANGE_BREAK", "HIT_35BP", "TO_35BP"),
    ("FIXED_10BP_DEPARTURE", "HIT_35BP", "TO_35BP_FROM_10BP"),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Phase 7E requires numpy and pandas. Activate the project .venv first."
        ) from exc
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:  # self-test and pure numerical helpers do not need parquet I/O
        pa = None
        pq = None
    return np, pd, pa, pq


def require_parquet_stack():
    np, pd, pa, pq = require_stack()
    if pa is None or pq is None:
        raise SystemExit(
            "Phase 7E Parquet I/O requires pyarrow. Activate the project .venv first."
        )
    return np, pd, pa, pq


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7e-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_parquet_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7e-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_phase_modules(args):
    phase6a = load_module(args.phase6a_script, "phase6a_frozen_for_7e")
    phase7c = load_module(args.phase7c_script, "phase7c_frozen_for_7e")
    phase7d1 = load_module(args.phase7d1_script, "phase7d1_frozen_for_7e")
    if getattr(phase6a, "OUTPUT_SCHEMA_NAME", None) != "perps_execution_foundation_v1":
        raise RuntimeError("Unexpected Phase-6A schema contract")
    if getattr(phase6a, "BUILDER_VERSION", None) != "1.0.4":
        raise RuntimeError(
            f"Phase 7E is frozen against the audited Phase-6A builder 1.0.4; found {getattr(phase6a, 'BUILDER_VERSION', None)!r}"
        )
    if getattr(phase7c, "SCHEMA_NAME", None) != "large_move_participant_behavior_atlas_v1":
        raise RuntimeError("Unexpected Phase-7C schema contract")
    if getattr(phase7c, "BUILDER_VERSION", None) != "1.0.2":
        raise RuntimeError(
            f"Phase 7E is frozen against Phase-7C builder 1.0.2; found {getattr(phase7c, 'BUILDER_VERSION', None)!r}"
        )
    if getattr(phase7d1, "SCHEMA_NAME", None) != "large_move_common_support_matched_controls_v1":
        raise RuntimeError("Unexpected Phase-7D.1 schema contract")
    if getattr(phase7d1, "BUILDER_VERSION", None) != "1.0.0":
        raise RuntimeError(
            f"Phase 7E is frozen against Phase-7D.1 builder 1.0.0; found {getattr(phase7d1, 'BUILDER_VERSION', None)!r}"
        )
    return phase6a, phase7c, phase7d1


def robust_scale(values: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return math.nan
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if math.isfinite(mad) and mad > 1e-12:
        return mad
    sd = float(np.std(a, ddof=1)) if len(a) >= 2 else math.nan
    return sd if math.isfinite(sd) and sd > 1e-12 else math.nan


def pooled_robust_scale(a: Sequence[float], b: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    x = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    return robust_scale(x)


def standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return math.nan
    sx = float(np.var(x, ddof=1))
    sy = float(np.var(y, ddof=1))
    pooled = math.sqrt(max(0.0, (sx + sy) / 2.0))
    if pooled <= 1e-12:
        return 0.0 if abs(float(np.mean(x)) - float(np.mean(y))) <= 1e-12 else math.nan
    return (float(np.mean(x)) - float(np.mean(y))) / pooled


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    np, _, _, _ = require_stack()
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return math.nan
    v = v[ok]
    w = w[ok]
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cutoff = float(w.sum()) / 2.0
    return float(v[np.searchsorted(np.cumsum(w), cutoff, side="left")])


def file_signature(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted(paths):
        st = p.stat()
        out.append({"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return out


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analysis_role_map() -> dict[str, str]:
    return {aid: role for role, aid in ANALYSIS_SPECS}


def parse_analysis_id(aid: str) -> dict[str, str]:
    parts = aid.split("|")
    if len(parts) != 4:
        raise RuntimeError(f"Unexpected analysis id: {aid}")
    return {"cohort": parts[0], "control_family": parts[1], "match_tier": parts[2], "caliper_policy": parts[3]}


def load_selected_assignments(args):
    _, pd, _, pq = require_stack()
    path = args.phase7d1_root / "control_assignments.parquet"
    if not path.exists():
        raise RuntimeError(f"Missing frozen Phase-7D.1 assignments: {path}")
    a = pq.read_table(path).to_pandas()
    wanted = {aid for _, aid in ANALYSIS_SPECS}
    a = a[a["analysis_id"].astype(str).isin(wanted)].copy()
    if "accepted_parent" in a.columns:
        a = a[a["accepted_parent"].fillna(False).astype(bool)].copy()
    missing = wanted - set(a["analysis_id"].astype(str).unique())
    if missing:
        raise RuntimeError(f"Missing required frozen 7D.1 analyses: {sorted(missing)}")
    if (a["control_family"].astype(str) != "THIRTYFIVE_ONLY").any():
        raise RuntimeError("Phase 7E selected analyses must use THIRTYFIVE_ONLY controls")
    if a["control_event_id"].isna().any():
        raise RuntimeError("Phase 7E requires physical control_event_id for every assignment")
    a["parent_event_id"] = a["parent_event_id"].astype(str)
    a["control_event_id"] = a["control_event_id"].astype(str)
    return a.reset_index(drop=True)


def recompute_analysis_quality(assignments):
    np, pd, _, _ = require_stack()
    rows = []
    for aid, g in assignments.groupby("analysis_id", sort=True):
        meta = parse_analysis_id(str(aid))
        match_features = sorted(
            c[len("event__"):]
            for c in g.columns
            if c.startswith("event__") and f"control__{c[len('event__'):]}" in g.columns
        )
        pp_rows = []
        for pid, pg in g.groupby("parent_event_id", sort=False):
            rec = {"parent_event_id": str(pid)}
            for f in match_features:
                rec[f"event__{f}"] = safe_float(pg[f"event__{f}"].iloc[0])
                cv = pd.to_numeric(pg[f"control__{f}"], errors="coerce")
                rec[f"control__{f}"] = float(cv.mean()) if cv.notna().any() else math.nan
            pp_rows.append(rec)
        pp = pd.DataFrame(pp_rows)
        abs_smd = []
        for f in match_features:
            x = pd.to_numeric(pp[f"event__{f}"], errors="coerce")
            y = pd.to_numeric(pp[f"control__{f}"], errors="coerce")
            ok = x.notna() & y.notna()
            smd = standardized_mean_difference(x[ok].to_numpy(float), y[ok].to_numpy(float))
            if math.isfinite(smd):
                abs_smd.append(abs(smd))
        n = int(g["parent_event_id"].nunique())
        max_smd = max(abs_smd) if abs_smd else math.nan
        grade = (
            "INTERPRETABLE_BALANCED"
            if n >= MIN_INTERPRETABLE_PARENTS and math.isfinite(max_smd) and max_smd <= BALANCE_GOOD_MAX_ABS_SMD
            else ("BALANCED_LIMITED_N" if math.isfinite(max_smd) and max_smd <= BALANCE_GOOD_MAX_ABS_SMD else "DO_NOT_INTERPRET")
        )
        rows.append({
            "analysis_id": aid,
            "analysis_role": analysis_role_map().get(str(aid), "OTHER"),
            **meta,
            "accepted_parents": n,
            "control_assignments": int(len(g)),
            "unique_controls": int(g["control_event_id"].nunique()),
            "match_features": ",".join(match_features),
            "max_abs_smd_recomputed": max_smd,
            "balance_grade_recomputed": grade,
        })
    q = pd.DataFrame(rows)
    primary = q[q["analysis_id"] == PRIMARY_ANALYSIS_ID]
    if len(primary) != 1:
        raise RuntimeError("Could not resolve unique primary Phase-7D.1 analysis")
    pr = primary.iloc[0]
    if pr["balance_grade_recomputed"] != "INTERPRETABLE_BALANCED":
        raise RuntimeError(
            "Primary 7E population no longer passes frozen balance gate: "
            f"parents={pr['accepted_parents']} max|SMD|={pr['max_abs_smd_recomputed']}"
        )
    return q.sort_values("analysis_id", kind="mergesort").reset_index(drop=True)


def build_event_stage_table(args, phase7c, assignments):
    _, pd, _, pq = require_stack()
    event_path = args.catalog_root / "physical_events.parquet"
    onset_path = args.catalog_root / "onset_candidates.parquet"
    if not event_path.exists() or not onset_path.exists():
        raise RuntimeError(f"Missing Phase-7A catalog artifacts: {event_path} / {onset_path}")
    events = pq.read_table(event_path).to_pandas()
    onsets = pq.read_table(onset_path).to_pandas()
    if "symbol" in events.columns:
        events = events[events["symbol"].astype(str) == args.symbol].copy()
    event_meta = phase7c.build_event_metadata(events).copy()
    event_meta["event_id"] = event_meta["event_id"].astype(str)
    needed = set(assignments["parent_event_id"]) | set(assignments["control_event_id"])
    em = event_meta[event_meta["event_id"].isin(needed)].copy()
    missing = needed - set(em["event_id"])
    if missing:
        raise RuntimeError(f"Missing selected event ids from catalog; first={sorted(missing)[:5]}")
    onsets["event_id"] = onsets["event_id"].astype(str)
    onsets = onsets[onsets["event_id"].isin(needed)].copy()
    onset_by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for eid, g in onsets.groupby("event_id", sort=False):
        onset_by_event[str(eid)] = {str(r["onset_method"]): r for r in g.to_dict("records")}
    rows = []
    for ev in em.to_dict("records"):
        stages = phase7c.stage_seconds_for_event(ev, onset_by_event.get(str(ev["event_id"]), {}))
        row = {
            "event_id": str(ev["event_id"]),
            "onset_date": str(ev["onset_date"]),
            "direction": int(ev["direction"]),
            "continuation_stratum": str(ev["continuation_stratum"]),
            "onset_second": int(ev["onset_second"]),
        }
        for st in STAGES:
            row[f"stage__{st}"] = stages.get(st)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out["event_id"].duplicated().any():
        raise RuntimeError("Selected event-stage table is not unique by event_id")
    return out.sort_values(["onset_date", "onset_second", "event_id"], kind="mergesort").reset_index(drop=True)


def selected_event_membership(assignments):
    _, pd, _, _ = require_stack()
    rows = []
    for aid, g in assignments.groupby("analysis_id", sort=True):
        role = analysis_role_map().get(str(aid), "OTHER")
        for eid in sorted(set(g["parent_event_id"])):
            rows.append({"analysis_id": aid, "analysis_role": role, "event_id": eid, "sample_role": "EVENT"})
        for eid in sorted(set(g["control_event_id"])):
            rows.append({"analysis_id": aid, "analysis_role": role, "event_id": eid, "sample_role": "CONTROL"})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def side_qty(side: Any, tick: int) -> float:
    try:
        q = float(side.qty[int(tick)])
    except Exception:
        return 0.0
    return q if math.isfinite(q) and q > 0 else 0.0


def book_mid(book: Any) -> float:
    bp, _ = book.bids.best()
    ap, _ = book.asks.best()
    if bp is None or ap is None or not (bp > 0 and ap > bp):
        return math.nan
    return (float(bp) + float(ap)) / 2.0


def role_side(direction: int, role: str) -> str:
    # UP: asks oppose the move, bids support it. DOWN: bids oppose, asks support.
    if role == "OPPOSING":
        return "ASK" if int(direction) > 0 else "BID"
    if role == "SUPPORT":
        return "BID" if int(direction) > 0 else "ASK"
    raise ValueError(role)


def role_distance_bps(price: float, mid0: float, direction: int, role: str) -> float:
    signed = (float(price) / float(mid0) - 1.0) * 10_000.0
    return float(direction) * signed if role == "OPPOSING" else -float(direction) * signed


def frozen_levels(book: Any, direction: int, role: str, mid0: float, price_scale: float, band_bps: float = FROZEN_NEAR_BPS) -> dict[int, float]:
    side_name = role_side(direction, role)
    side = book.asks if side_name == "ASK" else book.bids
    out: dict[int, float] = {}
    for price, qty in side.iter_best():
        dist = role_distance_bps(price, mid0, direction, role)
        if dist < -1e-9:
            continue
        if dist > band_bps + 1e-9:
            break
        tick = int(round(float(price) * float(price_scale)))
        if qty > 0:
            out[tick] = float(qty)
    return out


@dataclass
class LevelState:
    tick: int
    price: float
    initial_qty: float
    min_qty: float
    first_depletion_ns: int | None = None
    first_refill_ns: int | None = None
    gross_depletion_quote: float = 0.0
    gross_positive_quote: float = 0.0
    refill_after_depletion_quote: float = 0.0


@dataclass
class RoleTracker:
    role: str
    side_name: str
    initial_levels: dict[int, LevelState]
    initial_quote: float
    noninitial_near_add_quote: float = 0.0
    noninitial_near_new_level_add_quote: float = 0.0
    outer_add_quote: float = 0.0
    outer_new_level_add_quote: float = 0.0
    through_anchor_mid_add_quote: float = 0.0


@dataclass
class Endpoint:
    label: str
    end_ns: int
    end_second: int
    window_type: str
    transition_end_stage: str | None = None


@dataclass
class MechanicsTracker:
    event_id: str
    onset_date: str
    direction: int
    continuation_stratum: str
    stage: str
    stage_second: int
    anchor_end_ns: int
    mid0: float
    spread_bps0: float
    endpoints: list[Endpoint]
    roles: dict[str, RoleTracker]
    price_scale: float
    max_book_age_ms: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    completed_labels: set[str] = field(default_factory=set)
    censored_ns: int | None = None
    censor_reason: str | None = None

    @property
    def done(self) -> bool:
        return len(self.completed_labels) >= len(self.endpoints)

    @property
    def max_end_ns(self) -> int:
        return max((e.end_ns for e in self.endpoints), default=self.anchor_end_ns)

    def on_changes(self, ts_ns: int, changes: Sequence[tuple[str, int, float, float]]) -> None:
        if self.censored_ns is not None or ts_ns <= self.anchor_end_ns:
            return
        for side_name, tick, old_qty, new_qty in changes:
            price = float(tick) / self.price_scale
            delta = float(new_qty) - float(old_qty)
            if abs(delta) <= 1e-15:
                continue
            for role, rt in self.roles.items():
                if side_name != rt.side_name:
                    continue
                st = rt.initial_levels.get(int(tick))
                if st is not None:
                    st.min_qty = min(st.min_qty, max(0.0, float(new_qty)))
                    if delta < 0:
                        loss_quote = -delta * price
                        st.gross_depletion_quote += loss_quote
                        if st.first_depletion_ns is None:
                            st.first_depletion_ns = int(ts_ns)
                    else:
                        gain_quote = delta * price
                        st.gross_positive_quote += gain_quote
                        if st.first_depletion_ns is not None:
                            st.refill_after_depletion_quote += gain_quote
                            if st.first_refill_ns is None:
                                st.first_refill_ns = int(ts_ns)
                    continue

                # Price levels absent from the frozen <=5-bp cohort are kept
                # separate.  Their additions are location proxies, not same-order
                # replenishment evidence.
                if delta <= 0:
                    continue
                dist = role_distance_bps(price, self.mid0, self.direction, role)
                add_quote = delta * price
                if dist < -1e-9:
                    # The current side has translated through the stage-start
                    # mid.  Keep this separate from restoration of the frozen
                    # near-touch region.
                    rt.through_anchor_mid_add_quote += add_quote
                elif dist <= FROZEN_NEAR_BPS + 1e-9:
                    rt.noninitial_near_add_quote += add_quote
                    if old_qty <= 0:
                        rt.noninitial_near_new_level_add_quote += add_quote
                elif dist <= OUTER_BAND_BPS + 1e-9:
                    rt.outer_add_quote += add_quote
                    if old_qty <= 0:
                        rt.outer_new_level_add_quote += add_quote

    def censor(self, ts_ns: int, reason: str) -> None:
        if self.censored_ns is None:
            self.censored_ns = int(ts_ns)
            self.censor_reason = str(reason)

    def _role_metrics(self, book: Any, rt: RoleTracker, elapsed_sec: float) -> dict[str, float]:
        initial_quote = float(rt.initial_quote)
        side = book.asks if rt.side_name == "ASK" else book.bids
        current_quote = 0.0
        retained_quote = 0.0
        fully_deleted_initial_quote = 0.0
        loss_from_initial_quote = 0.0
        durable_recovery_quote = 0.0
        gross_depletion = 0.0
        gross_positive = 0.0
        refill_after = 0.0
        depleted_levels = 0
        refilled_levels = 0
        refill_latencies = []
        refill_weights = []
        surviving_levels = 0
        for st in rt.initial_levels.values():
            q = side_qty(side, st.tick)
            iq = st.initial_qty * st.price
            current_quote += q * st.price
            retained_quote += min(q, st.initial_qty) * st.price
            if q > 0:
                surviving_levels += 1
            else:
                fully_deleted_initial_quote += iq
            loss = max(0.0, st.initial_qty - st.min_qty) * st.price
            recovery = min(loss, max(0.0, q - st.min_qty) * st.price)
            loss_from_initial_quote += loss
            durable_recovery_quote += recovery
            gross_depletion += st.gross_depletion_quote
            gross_positive += st.gross_positive_quote
            refill_after += st.refill_after_depletion_quote
            if st.first_depletion_ns is not None:
                depleted_levels += 1
                if st.first_refill_ns is not None:
                    refilled_levels += 1
                    refill_latencies.append((st.first_refill_ns - st.first_depletion_ns) / 1_000_000.0)
                    refill_weights.append(max(iq, 1e-12))

        nlevels = len(rt.initial_levels)
        def div(a: float, b: float) -> float:
            return float(a / b) if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-12 else math.nan

        return {
            "initial_quote_5bps": initial_quote,
            "initial_level_count_5bps": float(nlevels),
            "same_price_current_quote": current_quote,
            "same_price_current_to_initial_ratio": div(current_quote, initial_quote),
            "retained_initial_quote_ratio": div(retained_quote, initial_quote),
            "fully_deleted_initial_quote_share": div(fully_deleted_initial_quote, initial_quote),
            "surviving_initial_level_share": div(float(surviving_levels), float(nlevels)),
            "gross_depletion_quote": gross_depletion,
            "gross_positive_update_quote": gross_positive,
            "same_price_refill_after_depletion_quote": refill_after,
            "gross_depletion_per_initial_quote": div(gross_depletion, initial_quote),
            "gross_depletion_quote_per_sec": div(gross_depletion, elapsed_sec),
            "same_price_refill_to_depletion_ratio": div(refill_after, gross_depletion),
            "same_price_refill_quote_per_sec": div(refill_after, elapsed_sec),
            "peak_initial_queue_loss_quote": loss_from_initial_quote,
            "peak_initial_queue_loss_share": div(loss_from_initial_quote, initial_quote),
            "durable_same_price_refill_quote": durable_recovery_quote,
            "durable_same_price_refill_fraction": div(durable_recovery_quote, loss_from_initial_quote),
            "durable_unrecovered_loss_quote": max(0.0, loss_from_initial_quote - durable_recovery_quote),
            "durable_unrecovered_loss_share_initial": div(max(0.0, loss_from_initial_quote - durable_recovery_quote), initial_quote),
            "depleted_initial_level_share": div(float(depleted_levels), float(nlevels)),
            "refilled_depleted_level_share": div(float(refilled_levels), float(depleted_levels)),
            "weighted_median_first_refill_latency_ms": weighted_median(refill_latencies, refill_weights),
            "noninitial_near_add_quote": float(rt.noninitial_near_add_quote),
            "noninitial_near_new_level_add_quote": float(rt.noninitial_near_new_level_add_quote),
            "outer_5_10bps_add_quote": float(rt.outer_add_quote),
            "outer_5_10bps_new_level_add_quote": float(rt.outer_new_level_add_quote),
            "through_anchor_mid_add_quote": float(rt.through_anchor_mid_add_quote),
            "near_restoration_to_depletion_proxy": div(refill_after + rt.noninitial_near_add_quote, gross_depletion),
            "outward_add_to_depletion_proxy": div(rt.outer_add_quote, gross_depletion),
            "outward_new_level_add_to_depletion_proxy": div(rt.outer_new_level_add_quote, gross_depletion),
        }

    def finalize_endpoint(self, ep: Endpoint, book: Any, trade_context: "TradeContext") -> dict[str, Any]:
        elapsed = max(0.0, (ep.end_ns - self.anchor_end_ns) / 1_000_000_000.0)
        continuity_valid = self.censored_ns is None or self.censored_ns > ep.end_ns
        last_state = getattr(book, "last_state_ts_ns", None)
        age_ms = (ep.end_ns - int(last_state)) / 1_000_000.0 if last_state is not None else math.inf
        endpoint_state_valid = bool(getattr(book, "valid", False)) and not bool(getattr(book, "bridge_pending", False)) and age_ms <= self.max_book_age_ms and math.isfinite(book_mid(book))
        valid = bool(continuity_valid and endpoint_state_valid)
        invalid_reason = None
        if not continuity_valid:
            invalid_reason = self.censor_reason
        elif not endpoint_state_valid:
            invalid_reason = "ENDPOINT_BOOK_INVALID_OR_STALE"
        row: dict[str, Any] = {
            "builder_version": BUILDER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "onset_date": self.onset_date,
            "direction": self.direction,
            "direction_label": "UP" if self.direction > 0 else "DOWN",
            "continuation_stratum": self.continuation_stratum,
            "stage": self.stage,
            "stage_second": self.stage_second,
            "window_label": ep.label,
            "window_type": ep.window_type,
            "transition_end_stage": ep.transition_end_stage,
            "window_end_second": ep.end_second,
            "elapsed_sec": elapsed,
            "anchor_mid": self.mid0,
            "anchor_spread_bps": self.spread_bps0,
            "window_valid": bool(valid),
            "invalid_reason": invalid_reason,
            "endpoint_book_age_ms": age_ms,
        }
        if not valid:
            self.rows.append(row)
            self.completed_labels.add(ep.label)
            return row

        mid1 = book_mid(book)
        row["end_mid"] = mid1
        row["directional_mid_move_bps"] = (
            self.direction * (mid1 / self.mid0 - 1.0) * 10_000.0
            if math.isfinite(mid1) and self.mid0 > 0 else math.nan
        )
        for role, rt in self.roles.items():
            metrics = self._role_metrics(book, rt, elapsed)
            prefix = "opposing" if role == "OPPOSING" else "support"
            for k, v in metrics.items():
                row[f"{prefix}__{k}"] = v

        flow = trade_context.range_summary(self.stage_second, ep.end_second, self.direction)
        row.update(flow)
        directional_aggr = safe_float(row.get("directional_aggressive_quote"))
        for prefix in ("opposing", "support"):
            dep = safe_float(row.get(f"{prefix}__gross_depletion_quote"))
            unrecovered = safe_float(row.get(f"{prefix}__durable_unrecovered_loss_quote"))
            row[f"{prefix}__gross_depletion_to_directional_aggression_ratio"] = (
                dep / directional_aggr if math.isfinite(dep) and math.isfinite(directional_aggr) and directional_aggr > 1e-12 else math.nan
            )
            row[f"{prefix}__unrecovered_loss_per_1m_directional_aggression"] = (
                unrecovered / (directional_aggr / 1_000_000.0)
                if math.isfinite(unrecovered) and math.isfinite(directional_aggr) and directional_aggr >= MIN_RESPONSE_NET_FLOW_QUOTE
                else math.nan
            )
        net = safe_float(row.get("directional_net_aggressive_quote"))
        move = safe_float(row.get("directional_mid_move_bps"))
        row["mid_move_bps_per_1m_net_aggressive_flow"] = (
            move / (net / 1_000_000.0)
            if math.isfinite(move) and math.isfinite(net) and abs(net) >= MIN_RESPONSE_NET_FLOW_QUOTE
            else math.nan
        )
        self.rows.append(row)
        self.completed_labels.add(ep.label)
        return row

    def finalize_due_before(self, ts_ns: int, book: Any, trade_context: "TradeContext") -> None:
        for ep in sorted(self.endpoints, key=lambda e: (e.end_ns, e.label)):
            if ep.label not in self.completed_labels and ep.end_ns < ts_ns:
                self.finalize_endpoint(ep, book, trade_context)

    def finalize_due_through(self, ts_ns: int, book: Any, trade_context: "TradeContext") -> None:
        for ep in sorted(self.endpoints, key=lambda e: (e.end_ns, e.label)):
            if ep.label not in self.completed_labels and ep.end_ns <= ts_ns:
                self.finalize_endpoint(ep, book, trade_context)

    def finalize_all(self, book: Any, trade_context: "TradeContext") -> None:
        for ep in sorted(self.endpoints, key=lambda e: (e.end_ns, e.label)):
            if ep.label not in self.completed_labels:
                if self.censored_ns is None:
                    self.censor(self.max_end_ns, "RAW_STREAM_ENDED_BEFORE_ENDPOINT")
                self.finalize_endpoint(ep, book, trade_context)

    def level_audit_rows(self, book: Any) -> list[dict[str, Any]]:
        rows = []
        for role, rt in self.roles.items():
            side = book.asks if rt.side_name == "ASK" else book.bids
            for st in rt.initial_levels.values():
                rows.append({
                    "builder_version": BUILDER_VERSION,
                    "event_id": self.event_id,
                    "onset_date": self.onset_date,
                    "direction": self.direction,
                    "continuation_stratum": self.continuation_stratum,
                    "stage": self.stage,
                    "stage_second": self.stage_second,
                    "side_role": role,
                    "book_side": rt.side_name,
                    "price": st.price,
                    "initial_qty": st.initial_qty,
                    "min_qty_observed": st.min_qty,
                    "final_qty_at_tracker_end": side_qty(side, st.tick),
                    "first_depletion_ns": st.first_depletion_ns,
                    "first_refill_ns": st.first_refill_ns,
                    "first_refill_latency_ms": (
                        (st.first_refill_ns - st.first_depletion_ns) / 1_000_000.0
                        if st.first_depletion_ns is not None and st.first_refill_ns is not None else math.nan
                    ),
                    "gross_depletion_quote": st.gross_depletion_quote,
                    "gross_positive_quote": st.gross_positive_quote,
                    "refill_after_depletion_quote": st.refill_after_depletion_quote,
                    "tracker_censor_reason": self.censor_reason,
                })
        return rows


class TradeContext:
    """Cumulative receive-second aggressive-trade context for raw windows."""

    def __init__(self, frame):
        np, pd, _, _ = require_stack()
        if frame is None or frame.empty:
            self.seconds = np.asarray([], dtype=np.int64)
            self.cums: dict[str, Any] = {}
            return
        f = frame.copy().sort_values("receive_second", kind="mergesort")
        self.seconds = pd.to_numeric(f["receive_second"], errors="coerce").fillna(-1).astype("int64").to_numpy()
        self.cums = {}
        for c in ("trade_count", "trade_quote_volume", "signed_aggressive_quote"):
            vals = pd.to_numeric(f.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
            self.cums[c] = np.concatenate([[0.0], np.cumsum(vals)])

    def _sum(self, column: str, start_exclusive: int, end_inclusive: int) -> float:
        if len(self.seconds) == 0 or end_inclusive <= start_exclusive:
            return 0.0
        lo = bisect.bisect_right(self.seconds, int(start_exclusive))
        hi = bisect.bisect_right(self.seconds, int(end_inclusive))
        c = self.cums[column]
        return float(c[hi] - c[lo])

    def range_summary(self, start_exclusive: int, end_inclusive: int, direction: int) -> dict[str, float]:
        total = self._sum("trade_quote_volume", start_exclusive, end_inclusive)
        signed = self._sum("signed_aggressive_quote", start_exclusive, end_inclusive)
        count = self._sum("trade_count", start_exclusive, end_inclusive)
        dnet = float(direction) * signed
        directional = max(0.0, (total + dnet) / 2.0)
        counter = max(0.0, (total - dnet) / 2.0)
        return {
            "forward_trade_count": count,
            "forward_trade_quote_volume": total,
            "forward_signed_aggressive_quote": signed,
            "directional_net_aggressive_quote": dnet,
            "directional_aggressive_quote": directional,
            "counter_direction_aggressive_quote": counter,
            "forward_directional_aggressive_imbalance": dnet / total if total > 1e-12 else math.nan,
        }


def spread_bps(book: Any) -> float:
    bp, _ = book.bids.best()
    ap, _ = book.asks.best()
    if bp is None or ap is None or not (ap > bp > 0):
        return math.nan
    mid = (bp + ap) / 2.0
    return (ap - bp) / mid * 10_000.0


def build_endpoints(stage: str, stage_second: int, stage_row: Mapping[str, Any]) -> list[Endpoint]:
    anchor_end_ns = (int(stage_second) + 1) * 1_000_000_000 - 1
    out = []
    for h in FIXED_HORIZONS_SEC:
        end_second = int(stage_second) + int(h)
        out.append(Endpoint(f"FWD_{h}S", anchor_end_ns + int(h) * 1_000_000_000, end_second, "FIXED_HORIZON"))
    for start_stage, end_stage, label in TRANSITIONS:
        if stage != start_stage:
            continue
        end_sec = stage_row.get(f"stage__{end_stage}")
        if end_sec is None or (isinstance(end_sec, float) and math.isnan(end_sec)):
            continue
        end_sec = int(end_sec)
        if end_sec <= int(stage_second):
            continue
        end_ns = (end_sec + 1) * 1_000_000_000 - 1
        out.append(Endpoint(label, end_ns, end_sec, "STAGE_TRANSITION", end_stage))
    # labels are unique by construction per start stage
    return sorted(out, key=lambda x: (x.end_ns, x.label))


def make_tracker(ev: Mapping[str, Any], stage: str, stage_second: int, book: Any, trade_context: TradeContext, max_age_ms: int, price_scale: float) -> tuple[MechanicsTracker | None, list[dict[str, Any]]]:
    anchor_end_ns = (int(stage_second) + 1) * 1_000_000_000 - 1
    age_ms = (
        (anchor_end_ns - int(book.last_state_ts_ns)) / 1_000_000.0
        if getattr(book, "last_state_ts_ns", None) is not None else math.inf
    )
    mid0 = book_mid(book)
    endpoints = build_endpoints(stage, int(stage_second), ev)
    if not endpoints:
        return None, []
    if not bool(getattr(book, "valid", False)) or bool(getattr(book, "bridge_pending", False)) or age_ms > max_age_ms or not math.isfinite(mid0):
        rows = []
        reason = "ANCHOR_BOOK_INVALID_OR_STALE"
        for ep in endpoints:
            rows.append({
                "builder_version": BUILDER_VERSION,
                "schema_version": SCHEMA_VERSION,
                "event_id": str(ev["event_id"]),
                "onset_date": str(ev["onset_date"]),
                "direction": int(ev["direction"]),
                "direction_label": "UP" if int(ev["direction"]) > 0 else "DOWN",
                "continuation_stratum": str(ev["continuation_stratum"]),
                "stage": stage,
                "stage_second": int(stage_second),
                "window_label": ep.label,
                "window_type": ep.window_type,
                "transition_end_stage": ep.transition_end_stage,
                "window_end_second": ep.end_second,
                "elapsed_sec": max(0.0, (ep.end_ns - anchor_end_ns) / 1_000_000_000.0),
                "anchor_mid": mid0,
                "anchor_spread_bps": spread_bps(book),
                "window_valid": False,
                "invalid_reason": reason,
            })
        return None, rows

    roles = {}
    for role in ("OPPOSING", "SUPPORT"):
        side_name = role_side(int(ev["direction"]), role)
        levels = frozen_levels(book, int(ev["direction"]), role, mid0, price_scale, FROZEN_NEAR_BPS)
        states = {
            tick: LevelState(tick, tick / price_scale, q, q)
            for tick, q in levels.items()
        }
        initial_quote = sum(st.price * st.initial_qty for st in states.values())
        roles[role] = RoleTracker(role, side_name, states, float(initial_quote))

    tr = MechanicsTracker(
        event_id=str(ev["event_id"]),
        onset_date=str(ev["onset_date"]),
        direction=int(ev["direction"]),
        continuation_stratum=str(ev["continuation_stratum"]),
        stage=stage,
        stage_second=int(stage_second),
        anchor_end_ns=anchor_end_ns,
        mid0=mid0,
        spread_bps0=spread_bps(book),
        endpoints=endpoints,
        roles=roles,
        price_scale=price_scale,
        max_book_age_ms=int(max_age_ms),
    )
    return tr, []


def extract_row_changes(book: Any, row: Mapping[str, Any]) -> list[tuple[str, int, float, float]]:
    """Read exact absolute-quantity changes BEFORE LocalBook mutates the row."""
    changes: list[tuple[str, int, float, float]] = []
    if "_fast_bid_ticks" in row:
        for side_name, side, ticks_key, qty_key in (
            ("BID", book.bids, "_fast_bid_ticks", "_fast_bid_qty"),
            ("ASK", book.asks, "_fast_ask_ticks", "_fast_ask_qty"),
        ):
            ticks = row.get(ticks_key)
            qtys = row.get(qty_key)
            if ticks is None or qtys is None:
                continue
            for tick, newq in zip(ticks, qtys):
                t = int(tick)
                oldq = side_qty(side, t)
                nq = max(0.0, float(newq))
                changes.append((side_name, t, oldq, nq))
        return changes

    for side_name, side, key in (("BID", book.bids, "bid_updates"), ("ASK", book.asks, "ask_updates")):
        for level in row.get(key) or []:
            if isinstance(level, Mapping):
                p = safe_float(level.get("price")); q = safe_float(level.get("quantity"))
            else:
                try:
                    p = safe_float(level[0]); q = safe_float(level[1])
                except Exception:
                    continue
            if not math.isfinite(p) or not math.isfinite(q):
                continue
            tick = int(round(p * book.price_scale))
            changes.append((side_name, tick, side_qty(side, tick), max(0.0, q)))
    return changes


def depth_row_is_applicable(book: Any, row: Mapping[str, Any]) -> bool:
    """True only when a depth row will be authoritative on an already-valid replay.

    Phase 7E anchors only on valid states. Bootstrap rows occurring before an
    anchor are handled by Phase-6A LocalBook but are not mechanism observations.
    """
    if not bool(getattr(book, "valid", False)) or bool(getattr(book, "bridge_pending", False)):
        return False
    prev = getattr(book, "previous_u", None)
    try:
        pu = int(row["previous_final_update_id"])
    except Exception:
        return False
    return prev is not None and pu == int(prev)


def build_day_files(phase6a, family_maps: Mapping[str, Mapping[date, tuple[Path, ...]]], days: Sequence[date]):
    def cat(family: str) -> tuple[Path, ...]:
        out: list[Path] = []
        for d in days:
            out.extend(family_maps[family].get(d, ()))
        return tuple(out)
    return phase6a.DayFiles(
        day=days[0],
        depth=cat("depth_updates"),
        snapshots=cat("book_snapshots"),
        trades=cat("trades"),
        liquidations=(),
    )


def day_partition_paths(output_root: Path, onset_date: str) -> tuple[Path, Path, Path]:
    root = output_root / "event_windows" / f"onset_date={onset_date}"
    level_root = output_root / "level_audit" / f"onset_date={onset_date}"
    return root / "part.parquet", level_root / "part.parquet", root / "manifest.json"


def day_input_signature(day_events, day_files, onset_date: str, builder_version: str = BUILDER_VERSION) -> str:
    payload = {
        "builder_version": str(builder_version),
        "schema_name": SCHEMA_NAME,
        "onset_date": onset_date,
        "events": day_events.sort_values("event_id", kind="mergesort").to_dict("records"),
        "canonical_files": file_signature(day_files.depth + day_files.snapshots + day_files.trades),
        "fixed_horizons_sec": FIXED_HORIZONS_SEC,
        "near_bps": FROZEN_NEAR_BPS,
        "outer_bps": OUTER_BAND_BPS,
    }
    return sha256_json(payload)


def process_onset_date(onset_date: str, day_events, phase6a, family_maps, args):
    np, pd, _, pq = require_stack()
    base = date.fromisoformat(onset_date)
    stage_seconds = []
    for r in day_events.to_dict("records"):
        for st in STAGES:
            v = r.get(f"stage__{st}")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                stage_seconds.append(int(v))
    if not stage_seconds:
        return pd.DataFrame(), pd.DataFrame(), False, {}
    max_stage = max(stage_seconds)
    max_end_second = max_stage + max(FIXED_HORIZONS_SEC)
    # Transition windows can end at HIT_35BP.
    if "stage__HIT_35BP" in day_events:
        hit = pd.to_numeric(day_events["stage__HIT_35BP"], errors="coerce").dropna()
        if len(hit):
            max_end_second = max(max_end_second, int(hit.max()))
    end_date = datetime.fromtimestamp(max_end_second, tz=timezone.utc).date()
    scan_days = [base]
    if end_date > base:
        scan_days.append(base + timedelta(days=1))
    day_files = build_day_files(phase6a, family_maps, scan_days)
    if not day_files.depth or not day_files.snapshots:
        raise RuntimeError(f"Missing canonical depth/snapshot files needed for onset date {onset_date}")

    profile_path, level_path, manifest_path = day_partition_paths(args.output_root, onset_date)
    sig = day_input_signature(day_events, day_files, onset_date)
    # v1.0.1-v1.0.3 are performance/runner-safety patches only.  They do not
    # change any per-day scientific profile or frozen-level measurement, so
    # completed checkpoints from earlier 7E builder versions remain reusable.
    legacy_sig_v100 = day_input_signature(day_events, day_files, onset_date, builder_version="1.0.0")
    legacy_sig_v101 = day_input_signature(day_events, day_files, onset_date, builder_version="1.0.1")
    legacy_sig_v102 = day_input_signature(day_events, day_files, onset_date, builder_version="1.0.2")
    if not args.force_day and profile_path.exists() and level_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        manifest_version = str(manifest.get("builder_version", ""))
        manifest_sig = manifest.get("input_signature")
        compatible = (
            (manifest_version == BUILDER_VERSION and manifest_sig == sig)
            or (manifest_version == "1.0.2" and manifest_sig == legacy_sig_v102)
            or (manifest_version == "1.0.1" and manifest_sig == legacy_sig_v101)
            or (manifest_version == "1.0.0" and manifest_sig == legacy_sig_v100)
        )
        if compatible:
            diag = {
                "trade_aggregation_seconds": manifest.get("trade_aggregation_seconds"),
                "raw_replay_seconds": manifest.get("raw_replay_seconds"),
                "depth_rows_decoded": manifest.get("depth_rows_decoded"),
                "depth_levels_decoded": manifest.get("depth_levels_decoded"),
                "depth_arrow_batches": manifest.get("depth_arrow_batches"),
                "tracked_depth_messages": manifest.get("tracked_depth_messages"),
                "tracked_exact_changes": manifest.get("tracked_exact_changes"),
            }
            return pq.read_table(profile_path).to_pandas(), pq.read_table(level_path).to_pandas(), True, diag

    # Raw aggressive trades are aggregated independently from depth updates.
    trade_t0 = time.perf_counter()
    trades = phase6a.aggregate_trades(day_files, args.symbol)
    trade_context = TradeContext(trades)
    trade_seconds = time.perf_counter() - trade_t0

    anchors: list[tuple[int, str, dict[str, Any]]] = []
    for ev in day_events.to_dict("records"):
        for st in STAGES:
            sec = ev.get(f"stage__{st}")
            if sec is None or (isinstance(sec, float) and math.isnan(sec)):
                continue
            sec = int(sec)
            anchors.append(((sec + 1) * 1_000_000_000 - 1, st, ev))
    anchors.sort(key=lambda x: (x[0], str(x[2]["event_id"]), x[1]))

    price_scale = float(args.price_scale)
    book = phase6a.LocalBook(price_scale=price_scale)
    fast_stats = phase6a.FastDepthStats()
    gap_records: list[Any] = []
    active: list[MechanicsTracker] = []
    finished: list[MechanicsTracker] = []
    profile_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    anchor_idx = 0
    stop_ns = (max_end_second + 1) * 1_000_000_000 - 1
    tracked_depth_messages = 0
    tracked_exact_changes = 0
    replay_t0 = time.perf_counter()

    def instantiate_anchors_before(ts_ns: int) -> None:
        nonlocal anchor_idx
        while anchor_idx < len(anchors) and anchors[anchor_idx][0] < ts_ns:
            _anchor_ns, st, ev = anchors[anchor_idx]
            sec = int(ev[f"stage__{st}"])
            tr, invalid = make_tracker(ev, st, sec, book, trade_context, args.max_book_age_ms, price_scale)
            profile_rows.extend(invalid)
            if tr is not None:
                active.append(tr)
            anchor_idx += 1

    for rec in phase6a.unified_book_events(day_files, args.symbol, price_scale, fast_stats):
        ts_ns = int(rec["ts_recv_ns"])
        if ts_ns > stop_ns:
            # The current book already contains every authoritative update at
            # or before stop_ns. Finalize due endpoints before terminating the
            # bounded scan; do not misclassify an intentional scan boundary as
            # missing raw data.
            instantiate_anchors_before(ts_ns)
            for tr in active:
                tr.finalize_due_before(ts_ns, book, trade_context)
            remain = []
            for tr in active:
                if tr.done:
                    finished.append(tr)
                    profile_rows.extend(tr.rows)
                    level_rows.extend(tr.level_audit_rows(book))
                else:
                    remain.append(tr)
            active = remain
            break

        instantiate_anchors_before(ts_ns)
        for tr in active:
            tr.finalize_due_before(ts_ns, book, trade_context)

        kind = rec.get("_kind")
        if kind == "SNAPSHOT":
            book.apply_snapshot(rec)
        elif kind == "DEPTH":
            applicable = depth_row_is_applicable(book, rec)
            # Performance-only gate: exact before/after quantity extraction is
            # required only while at least one live mechanism tracker can consume
            # the changes.  The LocalBook itself must still replay every
            # authoritative diff so sequence state and stage-anchor books remain
            # identical to v1.0.0.  Skipping Python-level change extraction while
            # no tracker is live does not alter any scientific output.
            track_changes = bool(active) and any(
                tr.censored_ns is None and not tr.done and ts_ns > tr.anchor_end_ns
                for tr in active
            )
            changes = extract_row_changes(book, rec) if applicable and track_changes else []
            if applicable and track_changes:
                tracked_depth_messages += 1
                tracked_exact_changes += len(changes)
            was_valid = bool(book.valid)
            previous_u = book.previous_u
            book.apply_depth(rec, gap_records, onset_date)
            still_valid = bool(book.valid)
            if applicable and was_valid and still_valid:
                for tr in active:
                    tr.on_changes(ts_ns, changes)
            elif was_valid and not still_valid:
                reason = gap_records[-1].reason if gap_records else "BOOK_INVALIDATED"
                for tr in active:
                    if ts_ns > tr.anchor_end_ns and not tr.done:
                        tr.censor(ts_ns, reason)

        for tr in active:
            tr.finalize_due_through(ts_ns, book, trade_context)
        remain = []
        for tr in active:
            if tr.done:
                finished.append(tr)
                profile_rows.extend(tr.rows)
                level_rows.extend(tr.level_audit_rows(book))
            else:
                remain.append(tr)
        active = remain

    # Finalize all endpoints that are covered by the bounded raw interval. Only
    # genuinely unobserved endpoints are censored.
    instantiate_anchors_before(stop_ns + 1)
    for tr in active:
        tr.finalize_due_through(stop_ns, book, trade_context)
        if not tr.done:
            tr.finalize_all(book, trade_context)
        finished.append(tr)
        profile_rows.extend(tr.rows)
        level_rows.extend(tr.level_audit_rows(book))
    active.clear()
    replay_seconds = time.perf_counter() - replay_t0

    profiles = pd.DataFrame(profile_rows)
    levels = pd.DataFrame(level_rows)
    if not profiles.empty:
        profiles["day_input_signature"] = sig
    if not levels.empty:
        levels["day_input_signature"] = sig
    write_parquet_atomic(profiles, profile_path)
    write_parquet_atomic(levels, level_path)
    atomic_write_json(manifest_path, {
        "builder_version": BUILDER_VERSION,
        "schema_name": SCHEMA_NAME,
        "onset_date": onset_date,
        "input_signature": sig,
        "event_count": int(day_events["event_id"].nunique()),
        "profile_rows": int(len(profiles)),
        "level_rows": int(len(levels)),
        "gap_records_observed_during_scan": int(len(gap_records)),
        "trade_aggregation_seconds": float(trade_seconds),
        "raw_replay_seconds": float(replay_seconds),
        "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0)),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0)),
        "depth_arrow_batches": int(getattr(fast_stats, "arrow_batches", 0)),
        "tracked_depth_messages": int(tracked_depth_messages),
        "tracked_exact_changes": int(tracked_exact_changes),
        "generated_at": utc_now_iso(),
    })
    diag = {
        "trade_aggregation_seconds": float(trade_seconds),
        "raw_replay_seconds": float(replay_seconds),
        "depth_rows_decoded": int(getattr(fast_stats, "rows_decoded", 0)),
        "depth_levels_decoded": int(getattr(fast_stats, "levels_decoded", 0)),
        "depth_arrow_batches": int(getattr(fast_stats, "arrow_batches", 0)),
        "tracked_depth_messages": int(tracked_depth_messages),
        "tracked_exact_changes": int(tracked_exact_changes),
    }
    return profiles, levels, False, diag


def numeric_profile_features(profiles) -> list[str]:
    _, pd, _, _ = require_stack()
    excluded = {
        "schema_version", "stage_second", "window_end_second", "elapsed_sec", "direction",
    }
    prefixes = (
        "opposing__", "support__",
    )
    explicit = {
        "directional_mid_move_bps", "forward_trade_count", "forward_trade_quote_volume",
        "forward_signed_aggressive_quote", "directional_net_aggressive_quote",
        "directional_aggressive_quote", "counter_direction_aggressive_quote",
        "forward_directional_aggressive_imbalance", "mid_move_bps_per_1m_net_aggressive_flow",
    }
    feats = []
    for c in profiles.columns:
        if c in excluded:
            continue
        if c.startswith(prefixes) or c in explicit:
            if pd.api.types.is_numeric_dtype(profiles[c]):
                feats.append(c)
    return sorted(feats)


def build_pair_rows(assignments, profiles, analysis_id: str, features: Sequence[str]):
    _, pd, _, _ = require_stack()
    a = assignments[assignments["analysis_id"] == analysis_id].copy()
    if a.empty:
        return pd.DataFrame()
    meta_cols = ["event_id", "onset_date", "direction", "stage", "window_label", "window_type", "transition_end_stage", "window_valid"]
    p = profiles[profiles["window_valid"].fillna(False).astype(bool)].copy()
    event_long = p[meta_cols + list(features)].rename(columns={"event_id": "parent_event_id", "onset_date": "parent_event_date"})
    ctrl_long = p[["event_id", "stage", "window_label"] + list(features)].rename(columns={"event_id": "control_event_id"})
    x = a[["parent_event_id", "control_event_id"]].merge(ctrl_long, on="control_event_id", how="inner")
    count = x.groupby(["parent_event_id", "stage", "window_label"], sort=False)["control_event_id"].nunique().rename("valid_control_count").reset_index()
    x = x.merge(count, on=["parent_event_id", "stage", "window_label"], how="left")
    x = x[x["valid_control_count"] >= MIN_VALID_CONTROLS_PER_PARENT].copy()
    control_mean = x.groupby(["parent_event_id", "stage", "window_label"], sort=False, as_index=False)[list(features)].mean()
    out = event_long.merge(control_mean, on=["parent_event_id", "stage", "window_label"], how="inner", suffixes=("__event", "__control"))
    rows = []
    for f in features:
        ec = f"{f}__event"; cc = f"{f}__control"
        if ec not in out or cc not in out:
            continue
        sub = out[["parent_event_id", "parent_event_date", "direction", "stage", "window_label", "window_type", "transition_end_stage", ec, cc]].copy()
        sub = sub.rename(columns={ec: "event_value", cc: "control_value"})
        sub["feature"] = f
        sub["paired_difference"] = sub["event_value"] - sub["control_value"]
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    z = pd.concat(rows, ignore_index=True)
    z["analysis_id"] = analysis_id
    z["analysis_role"] = analysis_role_map().get(analysis_id, "OTHER")
    return z


def summarize_pair_rows(pair_rows):
    np, pd, _, _ = require_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for (aid, role, st, win, wt, feat), g in pair_rows.groupby(
        ["analysis_id", "analysis_role", "stage", "window_label", "window_type", "feature"], sort=True
    ):
        ev = pd.to_numeric(g["event_value"], errors="coerce")
        co = pd.to_numeric(g["control_value"], errors="coerce")
        ok = ev.notna() & co.notna()
        ev = ev[ok].to_numpy(float); co = co[ok].to_numpy(float)
        if len(ev) == 0:
            continue
        diff = ev - co
        scale = pooled_robust_scale(ev, co)
        rows.append({
            "analysis_id": aid,
            "analysis_role": role,
            "stage": st,
            "window_label": win,
            "window_type": wt,
            "feature": feat,
            "parent_pairs": int(len(ev)),
            "event_median": float(np.median(ev)),
            "control_median": float(np.median(co)),
            "median_paired_difference": float(np.median(diff)),
            "paired_robust_standardized_difference": float(np.median(diff) / scale) if math.isfinite(scale) and scale > 1e-12 else math.nan,
            "paired_positive_share": float(np.mean(diff > 0)),
        })
    return pd.DataFrame(rows)


def daily_consistency(pair_rows):
    np, pd, _, _ = require_stack()
    rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for keys, g in pair_rows.groupby(["analysis_id", "stage", "window_label", "feature"], sort=True):
        daily = []
        for day, dg in g.groupby("parent_event_date", sort=True):
            d = pd.to_numeric(dg["paired_difference"], errors="coerce").dropna()
            if len(d) >= 2:
                daily.append((str(day), float(d.median())))
        if not daily:
            continue
        overall = pd.to_numeric(g["paired_difference"], errors="coerce").dropna()
        med = float(overall.median()) if len(overall) else math.nan
        sign = 1 if med > 0 else (-1 if med < 0 else 0)
        vals = np.asarray([x[1] for x in daily], dtype=float)
        rows.append({
            "analysis_id": keys[0], "stage": keys[1], "window_label": keys[2], "feature": keys[3],
            "days_comparable": int(len(daily)),
            "overall_parent_median_difference": med,
            "same_sign_day_share": float(np.mean(np.sign(vals) == sign)) if sign != 0 else math.nan,
            "first_day": daily[0][0], "last_day": daily[-1][0],
        })
    return pd.DataFrame(rows)


def subset_robustness(pair_rows, start: date, end: date):
    np, pd, _, _ = require_stack()
    if pair_rows.empty:
        return pd.DataFrame()
    midpoint = start + timedelta(days=(end - start).days // 2)
    work = pair_rows.copy()
    work["sample_half"] = work["parent_event_date"].astype(str).map(
        lambda x: "FIRST_HALF" if date.fromisoformat(x) <= midpoint else "SECOND_HALF"
    )
    rows = []
    for subset_type, col in (("DIRECTION", "direction"), ("SAMPLE_HALF", "sample_half")):
        for subset_value, sg in work.groupby(col, sort=True):
            for keys, g in sg.groupby(["analysis_id", "stage", "window_label", "feature"], sort=True):
                ev = pd.to_numeric(g["event_value"], errors="coerce")
                co = pd.to_numeric(g["control_value"], errors="coerce")
                ok = ev.notna() & co.notna()
                ev = ev[ok].to_numpy(float); co = co[ok].to_numpy(float)
                if len(ev) == 0:
                    continue
                diff = ev - co
                scale = pooled_robust_scale(ev, co)
                rows.append({
                    "analysis_id": keys[0], "stage": keys[1], "window_label": keys[2], "feature": keys[3],
                    "subset_type": subset_type,
                    "subset_value": ("UP" if int(subset_value) > 0 else "DOWN") if subset_type == "DIRECTION" else str(subset_value),
                    "parent_pairs": int(len(ev)),
                    "median_paired_difference": float(np.median(diff)),
                    "paired_robust_standardized_difference": float(np.median(diff) / scale) if math.isfinite(scale) and scale > 1e-12 else math.nan,
                })
    return pd.DataFrame(rows)


def build_profile_coverage(assignments, profiles):
    _, pd, _, _ = require_stack()
    rows = []
    valid = profiles[profiles["window_valid"].fillna(False).astype(bool)] if not profiles.empty else profiles
    for aid, ag in assignments.groupby("analysis_id", sort=True):
        parent_ids = set(ag["parent_event_id"])
        ctrl_ids = set(ag["control_event_id"])
        for st in STAGES:
            windows = sorted(set(profiles.loc[profiles["stage"] == st, "window_label"].astype(str))) if not profiles.empty else []
            for win in windows:
                pv = set(valid[(valid["stage"] == st) & (valid["window_label"] == win)]["event_id"].astype(str))
                p_good = parent_ids & pv
                c_good = ctrl_ids & pv
                # Preserve the original matched-control discipline: a parent
                # is raw-depth pairable only if the event profile is valid and
                # at least two assigned controls are valid at the same window.
                pairable = 0
                for pid, pg in ag.groupby("parent_event_id", sort=False):
                    if str(pid) not in pv:
                        continue
                    valid_controls = {str(x) for x in pg["control_event_id"] if str(x) in pv}
                    if len(valid_controls) >= MIN_VALID_CONTROLS_PER_PARENT:
                        pairable += 1
                rows.append({
                    "analysis_id": aid,
                    "analysis_role": analysis_role_map().get(str(aid), "OTHER"),
                    "stage": st,
                    "window_label": win,
                    "accepted_parents": int(len(parent_ids)),
                    "valid_parent_profiles": int(len(p_good)),
                    "valid_unique_control_profiles": int(len(c_good)),
                    "pairable_parents": int(pairable),
                    "pairable_parent_share": pairable / len(parent_ids) if parent_ids else math.nan,
                })
    return pd.DataFrame(rows)


def build_interpretable_screen(summary, daily, quality):
    np, pd, _, _ = require_stack()
    if summary.empty:
        return pd.DataFrame()
    q = quality[["analysis_id", "balance_grade_recomputed", "max_abs_smd_recomputed", "accepted_parents"]]
    out = summary.merge(q, on="analysis_id", how="left")
    if not daily.empty:
        out = out.merge(
            daily[["analysis_id", "stage", "window_label", "feature", "days_comparable", "same_sign_day_share"]],
            on=["analysis_id", "stage", "window_label", "feature"], how="left"
        )
    rsd = pd.to_numeric(out["paired_robust_standardized_difference"], errors="coerce")
    day = pd.to_numeric(out.get("same_sign_day_share"), errors="coerce")
    n = pd.to_numeric(out["parent_pairs"], errors="coerce")
    good_balance = out["balance_grade_recomputed"] == "INTERPRETABLE_BALANCED"
    robust = rsd.abs() >= 0.20
    enough = n >= 20
    day_ok = day.isna() | (day >= 0.65)
    out["descriptive_mechanism_screen"] = np.where(
        good_balance & enough & robust & day_ok,
        "BALANCED_DESCRIPTIVE_MECHANISM",
        np.where(good_balance, "BALANCED_WEAK_LIMITED_OR_INCONSISTENT", "DO_NOT_INTERPRET_MATCH_BALANCE"),
    )
    return out.sort_values(["descriptive_mechanism_screen", "analysis_role", "stage", "window_label", "feature"], kind="mergesort")


def check_invariants(assignments, quality, event_stages, profiles, level_audit_summary, coverage, partial_scope: bool = False):
    _, pd, _, _ = require_stack()
    rows = []
    def add(name: str, ok: bool, observed: Any, expected: Any, detail: str = ""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected, "detail": detail})

    analyses_present = sorted(assignments["analysis_id"].unique()) if not assignments.empty else []
    if partial_scope:
        add("selected_analyses_present", True, analyses_present, "not required in partial smoke scope", "Matched parent/control pairs can span onset dates outside the smoke subset.")
    else:
        add("selected_analyses_present", set(aid for _, aid in ANALYSIS_SPECS).issubset(set(analyses_present)), analyses_present, "all frozen 7E analyses")
    add("controls_are_35_only_physical_events", bool((assignments["control_family"] == "THIRTYFIVE_ONLY").all()), assignments["control_family"].value_counts().to_dict(), "THIRTYFIVE_ONLY only")
    add("control_event_ids_present", not assignments["control_event_id"].isna().any(), int(assignments["control_event_id"].isna().sum()), 0)
    add("event_stage_ids_unique", bool(event_stages["event_id"].is_unique), int(event_stages["event_id"].nunique()), len(event_stages))
    pr = quality[quality["analysis_id"] == PRIMARY_ANALYSIS_ID].iloc[0]
    add("primary_match_balance_frozen_good", pr["balance_grade_recomputed"] == "INTERPRETABLE_BALANCED", f"N={pr['accepted_parents']} max|SMD|={pr['max_abs_smd_recomputed']:.4f}", f"N>={MIN_INTERPRETABLE_PARENTS}, max|SMD|<={BALANCE_GOOD_MAX_ABS_SMD}")
    if not profiles.empty:
        key = ["event_id", "stage", "window_label"]
        dup = int(profiles.duplicated(key).sum())
        add("profile_keys_unique", dup == 0, dup, 0)
        valid = profiles[profiles["window_valid"].fillna(False).astype(bool)]
        for c in ("opposing__retained_initial_quote_ratio", "support__retained_initial_quote_ratio", "opposing__durable_same_price_refill_fraction", "support__durable_same_price_refill_fraction"):
            if c in valid:
                x = pd.to_numeric(valid[c], errors="coerce").dropna()
                bad = int(((x < -1e-9) | (x > 1 + 1e-9)).sum())
                add(f"bounded_{c}", bad == 0, bad, 0)
    else:
        add("profiles_nonempty", False, 0, ">0")
    level_rows = int(level_audit_summary.get("row_count", 0))
    if level_rows > 0:
        bad = int(level_audit_summary.get("nonpositive_initial_qty", 0))
        roles = sorted(str(x) for x in level_audit_summary.get("roles", set()))
        add("frozen_level_initial_qty_positive", bad == 0, bad, 0)
        add("level_roles_known", set(roles).issubset({"OPPOSING", "SUPPORT"}), roles, "OPPOSING/SUPPORT")
    else:
        add("level_audit_nonempty", False, 0, ">0")
    primary_cov = coverage[coverage["analysis_id"] == PRIMARY_ANALYSIS_ID] if not coverage.empty else pd.DataFrame()
    primary_pairable = int(primary_cov["pairable_parents"].max()) if not primary_cov.empty else 0
    if partial_scope:
        add("primary_has_pairable_raw_depth_windows", True, primary_pairable, "not required in partial smoke scope", "A partial onset-date subset may contain a parent without its matched control dates, or vice versa.")
    else:
        add("primary_has_pairable_raw_depth_windows", primary_pairable > 0, primary_pairable, ">0")
    return pd.DataFrame(rows)


def build_findings(summary_meta: Mapping[str, Any], quality, coverage, screen) -> str:
    lines = [
        "# Phase 7E — Raw-depth continuation mechanics",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{summary_meta['status']}**",
        "- Role: **RETROSPECTIVE RAW-DEPTH MECHANISM STUDY — NOT PROSPECTIVE ALPHA**",
        f"- Unique selected physical events profiled: **{summary_meta['selected_events']:,}**",
        f"- Raw-depth window profile rows: **{summary_meta['profile_rows']:,}**",
        f"- Frozen price-level audit rows: **{summary_meta['level_rows']:,}**",
        "",
        "## Frozen matched populations",
        "",
    ]
    for r in quality.sort_values(["analysis_role", "analysis_id"]).to_dict("records"):
        lines.append(
            f"- **{r['analysis_role']}** `{r['analysis_id']}`: parents={int(r['accepted_parents'])}, "
            f"controls={int(r['unique_controls'])}, recomputed max|SMD|={safe_float(r['max_abs_smd_recomputed']):.3f}, "
            f"grade={r['balance_grade_recomputed']}."
        )
    lines += [
        "",
        "## Measurement contract",
        "",
        "- A stage anchor is the end of the causal receive-second representing that hindsight stage.",
        "- Exact price levels visible within 5 bps at the anchor are frozen and followed at the same prices.",
        "- Same-price refill is exact price-level quantity recovery, but participant/order identity is not observed.",
        "- Depth decreases are not labelled as cancellations versus trades. Aggregate raw aggressive trade pressure is reported separately.",
        "- 5-10-bp additions are an outward-replacement proxy, not proof of cancel/repost by the same participant.",
        "",
        "## Raw-depth matched coverage",
        "",
    ]
    pc = coverage[coverage["analysis_id"] == PRIMARY_ANALYSIS_ID] if not coverage.empty else coverage
    for r in pc.sort_values(["stage", "window_label"]).to_dict("records")[:40]:
        lines.append(
            f"- {r['stage']} / {r['window_label']}: pairable parents "
            f"{int(r['pairable_parents'])}/{int(r['accepted_parents'])} ({100*safe_float(r['pairable_parent_share']):.1f}%)."
        )
    lines += ["", "## Primary balanced descriptive mechanism screen", ""]
    if screen.empty:
        lines.append("- No mechanism rows were available.")
    else:
        good = screen[(screen["analysis_id"] == PRIMARY_ANALYSIS_ID) & (screen["descriptive_mechanism_screen"] == "BALANCED_DESCRIPTIVE_MECHANISM")].copy()
        if good.empty:
            lines.append("- No primary raw-depth feature passed the prespecified descriptive mechanism screen.")
        else:
            good["abs_rsd"] = good["paired_robust_standardized_difference"].abs()
            for r in good.sort_values("abs_rsd", ascending=False).head(30).to_dict("records"):
                lines.append(
                    f"- {r['stage']} / {r['window_label']} / `{r['feature']}`: "
                    f"event median={safe_float(r['event_median']):.6g}, control median={safe_float(r['control_median']):.6g}, "
                    f"paired RSD={safe_float(r['paired_robust_standardized_difference']):+.3f}, "
                    f"N={int(r['parent_pairs'])}, day-sign={safe_float(r.get('same_sign_day_share')):.1%}."
                )
    lines += [
        "",
        "## Interpretation discipline",
        "",
        "A matched residual here is a retrospective physical-move mechanism result. It is not automatically a causal precursor, a prospective detector, or a tradable edge. Phase 7E is intended to decide whether exact same-price refill/depletion/repricing mechanics deserve a later all-market prospective test.",
        "",
    ]
    return "\n".join(lines)


class _FakeSide:
    def __init__(self, levels: Mapping[int, float], scale: float, is_bid: bool):
        self.qty = dict((int(k), float(v)) for k, v in levels.items())
        self.scale = scale; self.is_bid = is_bid
    def best(self):
        keys = [k for k, v in self.qty.items() if v > 0]
        if not keys: return None, None
        k = max(keys) if self.is_bid else min(keys)
        return k / self.scale, self.qty[k]
    def iter_best(self):
        keys = sorted((k for k, v in self.qty.items() if v > 0), reverse=self.is_bid)
        for k in keys: yield k / self.scale, self.qty[k]


class _FakeBook:
    def __init__(self):
        self.price_scale = 100.0
        self.bids = _FakeSide({9999: 2.0, 9998: 1.0}, 100.0, True)
        self.asks = _FakeSide({10001: 2.0, 10002: 1.0}, 100.0, False)
        self.valid = True; self.bridge_pending = False; self.last_state_ts_ns = 1_000_000_000; self.previous_u = 1


def self_test() -> None:
    np, pd, _, _ = require_stack()
    book = _FakeBook()
    mid = book_mid(book)
    assert abs(mid - 100.0) < 1e-12
    opp = frozen_levels(book, +1, "OPPOSING", mid, 100.0, 5.0)
    assert 10001 in opp and 10002 in opp
    ev = {
        "event_id": "E1", "onset_date": "2026-01-01", "direction": 1, "continuation_stratum": "75_ONLY",
        "stage__FIXED_10BP_DEPARTURE": 1, "stage__PRIOR_60S_RANGE_BREAK": 4, "stage__HIT_35BP": 8,
    }
    tr, invalid = make_tracker(ev, "FIXED_10BP_DEPARTURE", 1, book, TradeContext(pd.DataFrame()), 2000, 100.0)
    assert tr is not None and not invalid
    # Deplete then refill the best ask at the exact same price.
    tr.on_changes(2_100_000_000, [("ASK", 10001, 2.0, 1.0)])
    book.asks.qty[10001] = 1.0
    tr.on_changes(2_300_000_000, [("ASK", 10001, 1.0, 1.5)])
    book.asks.qty[10001] = 1.5
    ep = next(e for e in tr.endpoints if e.label == "FWD_1S")
    row = tr.finalize_endpoint(ep, book, TradeContext(pd.DataFrame()))
    assert row["window_valid"]
    assert row["opposing__gross_depletion_quote"] > 0
    assert row["opposing__same_price_refill_after_depletion_quote"] > 0
    assert 0 <= row["opposing__durable_same_price_refill_fraction"] <= 1
    assert abs(row["opposing__weighted_median_first_refill_latency_ms"] - 200.0) < 1e-9
    # Weighted median and trade-direction arithmetic.
    assert weighted_median([1, 5], [10, 1]) == 1
    tc = TradeContext(pd.DataFrame({"receive_second": [2, 3], "trade_count": [2, 3], "trade_quote_volume": [100.0, 200.0], "signed_aggressive_quote": [60.0, -20.0]}))
    s = tc.range_summary(1, 3, +1)
    assert abs(s["forward_trade_quote_volume"] - 300.0) < 1e-9
    assert abs(s["directional_net_aggressive_quote"] - 40.0) < 1e-9
    print("SELF-TEST PASS")
    print("Validated frozen exact-price cohorts, same-price depletion/refill, durable refill bounds, refill latency, stage endpoints and raw aggressive-flow arithmetic.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Build Phase-7E exact raw-depth continuation mechanics for matched physical moves")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    p.add_argument("--phase7d1-root", type=Path, default=DEFAULT_PHASE7D1_ROOT)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--phase7c-script", type=Path, default=DEFAULT_PHASE7C_SCRIPT)
    p.add_argument("--phase7d1-script", type=Path, default=DEFAULT_PHASE7D1_SCRIPT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--price-scale", type=float, default=PRICE_SCALE)
    p.add_argument("--max-book-age-ms", type=int, default=MAX_BOOK_AGE_MS)
    p.add_argument("--force-day", action="store_true", help="Rebuild per-onset-date raw replay checkpoints even when signatures match")
    p.add_argument("--max-onset-dates", type=int, default=None, help="Smoke-test only: process first N selected onset dates")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.price_scale <= 0:
        raise SystemExit("--price-scale must be positive")
    if args.max_book_age_ms <= 0:
        raise SystemExit("--max-book-age-ms must be positive")

    np, pd, _, pq = require_parquet_stack()
    phase6a, phase7c, phase7d1 = load_phase_modules(args)
    print("Building Phase 7E exact raw-depth continuation mechanics...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-6A      : {getattr(phase6a, 'BUILDER_VERSION', 'unknown')}")
    print(f"Frozen Phase-7C      : {getattr(phase7c, 'BUILDER_VERSION', 'unknown')}")
    print(f"Frozen Phase-7D.1    : {phase7d1.BUILDER_VERSION}")
    print(f"Symbol               : {args.symbol}")
    print(f"Primary analysis     : {PRIMARY_ANALYSIS_ID}")
    print("Research role        : retrospective exact depth mechanics; not prospective alpha")

    print("[1/8] Loading frozen common-support assignments and rechecking match balance...")
    assignments = load_selected_assignments(args)
    quality = recompute_analysis_quality(assignments)
    for r in quality.to_dict("records"):
        print(f"      {r['analysis_role']:<18} parents={int(r['accepted_parents']):3d} controls={int(r['unique_controls']):3d} max|SMD|={safe_float(r['max_abs_smd_recomputed']):.3f} {r['balance_grade_recomputed']}")

    print("[2/8] Reconstructing selected physical-event stage timestamps...")
    event_stages = build_event_stage_table(args, phase7c, assignments)
    membership = selected_event_membership(assignments)
    print(f"      unique selected physical events={len(event_stages):,}")
    print(f"      event-analysis memberships={len(membership):,}")

    print("[3/8] Discovering canonical raw-depth/trade partitions...")
    family_maps = {
        fam: phase6a.discover_family_days(args.canonical_root, fam)
        for fam in ("depth_updates", "book_snapshots", "trades")
    }
    all_selected_dates = sorted(event_stages["onset_date"].astype(str).unique())
    selected_dates = list(all_selected_dates)
    if args.max_onset_dates is not None:
        selected_dates = selected_dates[: max(0, int(args.max_onset_dates))]
    partial_scope = len(selected_dates) < len(all_selected_dates)
    print(f"      selected onset dates={len(selected_dates):,}" + (f" of {len(all_selected_dates):,} (PARTIAL SMOKE SCOPE)" if partial_scope else ""))
    print("      checkpointing is by onset_date; unchanged signatures are reused")

    print("[4/8] Replaying authoritative depth diffs and materializing exact-price mechanics...")
    profile_parts = []
    level_audit_summary = {"row_count": 0, "nonpositive_initial_qty": 0, "roles": set()}
    reused_days = 0
    _, _, pa, _ = require_stack()
    for i, d in enumerate(selected_dates, 1):
        de = event_stages[event_stages["onset_date"] == d].copy()
        profiles_day, levels_day, reused, diag = process_onset_date(d, de, phase6a, family_maps, args)
        profile_parts.append(profiles_day)

        # Frozen-level rows are already persisted atomically per onset date.
        # Keeping every day's wide audit DataFrame alive until the end of a
        # 60-day replay creates unnecessary cumulative RAM pressure.  Only the
        # two global invariants below require those rows after day processing,
        # so aggregate them here and release the daily frame immediately.
        if not levels_day.empty:
            level_audit_summary["row_count"] += int(len(levels_day))
            initial = pd.to_numeric(levels_day["initial_qty"], errors="coerce")
            level_audit_summary["nonpositive_initial_qty"] += int((initial <= 0).sum())
            level_audit_summary["roles"].update(str(x) for x in levels_day["side_role"].dropna().unique())

        reused_days += int(reused)
        valid = int(profiles_day["window_valid"].fillna(False).sum()) if not profiles_day.empty else 0
        line = f"      [{i:02d}/{len(selected_dates):02d}] {d}: events={len(de):3d}, windows={len(profiles_day):5d}, valid={valid:5d}, {'REUSED' if reused else 'BUILT'}"
        if not reused:
            line += (
                f" | trade={safe_float(diag.get('trade_aggregation_seconds')):.1f}s"
                f" replay={safe_float(diag.get('raw_replay_seconds')):.1f}s"
                f" depth_rows={int(diag.get('depth_rows_decoded') or 0):,}"
                f" levels={int(diag.get('depth_levels_decoded') or 0):,}"
                f" tracked_msgs={int(diag.get('tracked_depth_messages') or 0):,}"
            )
        print(line)

        # Force release of the per-day LocalBook/Arrow/Numba/Pandas object graph
        # before decoding the next very large nested depth partition.  Arrow's
        # allocator can retain freed slabs, so explicitly ask it to return
        # unused pages when the installed backend supports that operation.
        del levels_day, de
        gc.collect()
        if pa is not None:
            try:
                pa.default_memory_pool().release_unused()
            except Exception:
                pass

    profiles = pd.concat(profile_parts, ignore_index=True) if profile_parts else pd.DataFrame()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(event_stages[event_stages["onset_date"].isin(selected_dates)].copy(), args.output_root / "selected_event_stages.parquet")
    write_parquet_atomic(membership[membership["event_id"].isin(set(event_stages[event_stages["onset_date"].isin(selected_dates)]["event_id"]))].copy(), args.output_root / "selected_event_membership.parquet")

    print("[5/8] Building parent-paired raw-depth mechanism contrasts...")
    processed_ids = set(event_stages[event_stages["onset_date"].isin(selected_dates)]["event_id"].astype(str))
    a_proc = assignments[assignments["parent_event_id"].isin(processed_ids) & assignments["control_event_id"].isin(processed_ids)].copy()
    features = numeric_profile_features(profiles) if not profiles.empty else []
    pair_parts = [build_pair_rows(a_proc, profiles, aid, features) for _, aid in ANALYSIS_SPECS]
    pair_rows = pd.concat([x for x in pair_parts if not x.empty], ignore_index=True) if any(not x.empty for x in pair_parts) else pd.DataFrame()
    contrasts = summarize_pair_rows(pair_rows)
    daily = daily_consistency(pair_rows)
    if processed_ids:
        dates = sorted(date.fromisoformat(x) for x in event_stages[event_stages["event_id"].isin(processed_ids)]["onset_date"].unique())
        robust = subset_robustness(pair_rows, dates[0], dates[-1]) if dates else pd.DataFrame()
    else:
        robust = pd.DataFrame()
    coverage = build_profile_coverage(a_proc, profiles)
    screen = build_interpretable_screen(contrasts, daily, quality)

    print("[6/8] Checking raw replay/mechanism invariants...")
    checks = check_invariants(a_proc, quality, event_stages[event_stages["event_id"].isin(processed_ids)], profiles, level_audit_summary, coverage, partial_scope=partial_scope)
    failed = checks[checks["status"] == "FAIL"]
    for r in checks.to_dict("records"):
        print(f"      {r['status']:<4} {r['check']}: observed={r['observed']} expected={r['expected']}")

    print("[7/8] Writing reports and metadata...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.report_root / "raw_depth_analysis_quality.csv", index=False)
    coverage.to_csv(args.report_root / "raw_depth_profile_coverage.csv", index=False)
    contrasts.to_csv(args.report_root / "raw_depth_mechanism_contrasts.csv", index=False)
    daily.to_csv(args.report_root / "raw_depth_daily_consistency.csv", index=False)
    robust.to_csv(args.report_root / "raw_depth_subset_robustness.csv", index=False)
    screen.to_csv(args.report_root / "raw_depth_interpretable_screen.csv", index=False)
    checks.to_csv(args.report_root / "raw_depth_invariants.csv", index=False)

    summary = {
        "status": ("SMOKE_PASS" if partial_scope else "PASS") if failed.empty else "FAIL",
        "run_scope": "PARTIAL_SMOKE" if partial_scope else "FULL",
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "symbol": args.symbol,
        "research_role": "RETROSPECTIVE_RAW_DEPTH_MECHANISM_NOT_PROSPECTIVE_ALPHA",
        "primary_analysis_id": PRIMARY_ANALYSIS_ID,
        "selected_analysis_ids": [aid for _, aid in ANALYSIS_SPECS],
        "selected_events": int(len(processed_ids)),
        "selected_onset_dates": int(len(selected_dates)),
        "total_available_onset_dates": int(len(all_selected_dates)),
        "reused_onset_date_checkpoints": int(reused_days),
        "profile_rows": int(len(profiles)),
        "valid_profile_rows": int(profiles["window_valid"].fillna(False).sum()) if not profiles.empty else 0,
        "level_rows": int(level_audit_summary["row_count"]),
        "mechanism_features": int(len(features)),
        "pair_rows": int(len(pair_rows)),
        "contrast_rows": int(len(contrasts)),
        "failed_invariants": int(len(failed)),
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", {
        **summary,
        "canonical_root": str(args.canonical_root.resolve()),
        "catalog_root": str(args.catalog_root.resolve()),
        "phase7d1_root": str(args.phase7d1_root.resolve()),
        "phase6a_script": str(args.phase6a_script.resolve()),
        "phase7c_script": str(args.phase7c_script.resolve()),
        "phase7d1_script": str(args.phase7d1_script.resolve()),
        "price_scale": args.price_scale,
        "max_book_age_ms": args.max_book_age_ms,
        "fixed_horizons_sec": list(FIXED_HORIZONS_SEC),
        "frozen_near_bps": FROZEN_NEAR_BPS,
        "outer_band_bps": OUTER_BAND_BPS,
        "min_valid_controls_per_parent": MIN_VALID_CONTROLS_PER_PARENT,
        "transitions": list(TRANSITIONS),
        "interpretation_limits": [
            "hindsight event/stage/continuation labels",
            "depth decrease does not identify cancellation versus trade",
            "same-price refill lacks order/participant identity",
            "outward replacement is a price-level addition proxy",
            "no prospective or tradable-edge claim",
        ],
    })
    atomic_write_text(args.report_root / "raw_depth_findings.md", build_findings(summary, quality, coverage, screen))

    print("[8/8] Phase 7E summary")
    print(f"      status={summary['status']}")
    print(f"      selected events={summary['selected_events']:,}")
    print(f"      raw window rows={summary['profile_rows']:,} valid={summary['valid_profile_rows']:,}")
    print(f"      frozen level rows={summary['level_rows']:,}")
    print(f"      mechanism contrast rows={summary['contrast_rows']:,}")
    print(f"      reports={args.report_root}")
    if not failed.empty:
        print("ERROR: one or more Phase-7E invariants failed; do not interpret the mechanism reports.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
