#!/usr/bin/env python3
"""
Phase 7I.9 — Large-move ontology robustness audit
=================================================

Scientific purpose
------------------
Audit whether the frozen Phase-7A / Phase-7I large-move ontology is selecting
the economically meaningful BTCUSDT expansions we actually care about.

This phase is deliberately retrospective and does NOT build a trading model.
It independently scans the Phase-6A one-second execution-price substrate using
a fixed 10-second anchor grid, creates several alternate large-move definitions,
deduplicates overlapping detections into alternate physical event clusters, and
then compares those events with:

  * frozen Phase-7A 35-bp directional-change physical legs;
  * frozen Phase-7I.1 causal 35-bp candidates; and
  * frozen Phase-7I.6 3-second exact-book failure state where a causal candidate
    is available.

The key question is not whether our current ontology is "the one true"
definition.  The audit asks whether economically obvious alternate definitions
select broadly the same expansions, which events the current ontology misses,
and whether the book-failure mechanism survives alternate labels.

Alternate definitions
---------------------
Outcome definitions are predeclared and are NOT tuned on book-mechanics results.

1. FIRST_PASSAGE_75_5M
2. FIRST_PASSAGE_75_10M
3. FIRST_PASSAGE_75_15M
4. TERMINAL_75_5M
5. TERMINAL_75_10M
6. TERMINAL_75_15M
7. PATH_EFFICIENT_75_10M
      first passage to +75 bps within 10m with <=15 bps adverse excursion
      before the target
8. VOLNORM_3X_RV15_10M
      first passage within 10m to max(50 bps, 3 x trailing 15m realized vol)
9. FIRST_PASSAGE_100_10M

Independent-anchor contract
---------------------------
* Anchors are every 10 UTC seconds where the Phase-6A catalog price is valid.
* Fixed-horizon and path-efficient definitions use only the anchor price and
  future path; they do not depend on Phase-7A or Phase-7I candidate construction.
* Volatility-normalized thresholds use only trailing returns available at the
  anchor.
* Fixed-75/100/path-efficient definitions share ONE common parent impulse
  clustering based on the union of available FIRST_PASSAGE_75_5M/10M/15M
  anchors. Stricter definitions inherit the same impulse ID, so nested event
  counts are set-consistent even at coverage/gap boundaries.
* Common parent clusters use a bounded first-hit span rather than transitive
  chaining, preventing one long directional leg from swallowing arbitrarily
  distant re-accelerations.
* The current Phase-7A / Phase-7I ontologies are used only AFTER alternate
  detections are built.

Outputs
-------
data/derived/perps/research/causal_large_move_ontology_robustness_audit_v1/
    alternate_detections.parquet
    alternate_impulses.parquet
    alternate_events.parquet
    candidate_ontology_membership.parquet
    metadata/latest_summary.json
    metadata/run_config.json

reports/perps_large_move_ontology_robustness/
    ontology_definition_summary.csv
    ontology_event_coverage.csv
    ontology_candidate_membership_rates.csv
    ontology_overlap_matrix.csv
    ontology_nested_consistency.csv
    ontology_physical_leg_impulse_multiplicity.csv
    ontology_reacceleration_impulses.csv
    ontology_missed_fast_events.csv
    ontology_book_failure_robustness.csv
    ontology_book_failure_direction_robustness.csv
    ontology_invariants.csv
    ontology_findings.md

Interpretation discipline
-------------------------
* Alternate definitions are retrospective research labels, not live signals.
* The 10-second anchor cadence is an audit grid, not a trading cadence.
* Book mechanics are NOT replayed.  Book-failure robustness is measurable only
  for alternate events already covered by the frozen causal-candidate universe.
* A miss in the book-failure audit can therefore mean "no frozen candidate
  mechanics available", not "the alternate event had healthy liquidity".
* No profitability claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BUILDER_VERSION = "1.0.2"
SCHEMA_VERSION = 1
SCHEMA_NAME = "causal_large_move_ontology_robustness_audit_v1"
DEFAULT_SYMBOL = "BTCUSDT"

FROZEN_PHASE7A_VERSION = "1.0.0"
FROZEN_PHASE7I1_VERSION = "1.0.0"

ANCHOR_CADENCE_SEC = 10
FIXED_HORIZONS_SEC = (300, 600, 900)
FIRST_PASSAGE_TARGET_BPS = 75.0
FAST_100_TARGET_BPS = 100.0
PATH_EFFICIENT_HORIZON_SEC = 600
PATH_EFFICIENT_MAX_PREHIT_MAE_BPS = 15.0
VOL_LOOKBACK_SEC = 900
VOL_MULTIPLIER = 3.0
VOL_TARGET_FLOOR_BPS = 50.0
VOL_HORIZON_SEC = 600
CLUSTER_MAX_ANCHOR_GAP_SEC = 60
CLUSTER_MAX_HIT_GAP_SEC = 180
COMMON_PARENT_SOURCE_DEFINITIONS = (
    "FIRST_PASSAGE_75_5M",
    "FIRST_PASSAGE_75_10M",
    "FIRST_PASSAGE_75_15M",
)
COMMON_PARENT_DEFINITIONS = (
    "FIRST_PASSAGE_75_5M",
    "FIRST_PASSAGE_75_10M",
    "FIRST_PASSAGE_75_15M",
    "TERMINAL_75_5M",
    "TERMINAL_75_10M",
    "TERMINAL_75_15M",
    "PATH_EFFICIENT_75_10M",
    "FIRST_PASSAGE_100_10M",
)
BOOK_DECISION_HORIZON_SEC = 3
DEFAULT_MAX_TRADE_STALE_SEC = 5

DEFINITION_ORDER = (
    "FIRST_PASSAGE_75_5M",
    "FIRST_PASSAGE_75_10M",
    "FIRST_PASSAGE_75_15M",
    "TERMINAL_75_5M",
    "TERMINAL_75_10M",
    "TERMINAL_75_15M",
    "PATH_EFFICIENT_75_10M",
    "VOLNORM_3X_RV15_10M",
    "FIRST_PASSAGE_100_10M",
)
CURRENT_PRIMARY_DEFINITION = "CURRENT_7A_75_WITHIN_30M"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()

DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07a_build_large_move_event_catalog.py"
DEFAULT_PHASE7I1_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07i1_build_causal_large_move_candidate_catalog.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_PHASE7A_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_PHASE7I1_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "causal_large_move_candidate_catalog_v1"
DEFAULT_PHASE7I6_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "causal_candidate_participant_interaction_analysis_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_large_move_ontology_robustness"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: Any, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def iso_time(sec: int | float | None) -> str | None:
    if sec is None or not math.isfinite(safe_float(sec)):
        return None
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).replace(microsecond=0).isoformat()


def require_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "Phase 7I.9 requires numpy and pandas. Activate the project .venv first."
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
            "Phase 7I.9 Parquet I/O requires pyarrow. Activate the project .venv first."
        )
    return np, pd, pa, pq


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
    fd, tmp = tempfile.mkstemp(prefix="tmp7i9-", suffix=path.suffix, dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7i9-", suffix=".csv", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7i9-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def contiguous_segments(sec, price):
    """Yield inclusive [start,end] indices for contiguous valid one-second paths."""
    np, _, _, _ = require_stack()
    valid = np.isfinite(price) & (price > 0)
    start = None
    prev = None
    for i in range(len(sec)):
        if not valid[i]:
            if start is not None:
                yield start, prev
                start = None
            prev = None
            continue
        if start is None:
            start = i
        elif prev is None or int(sec[i]) != int(sec[prev]) + 1:
            yield start, prev
            start = i
        prev = i
    if start is not None and prev is not None:
        yield start, prev


def future_extreme_arrays(sec, price, horizons: Sequence[int]):
    """Forward max/min over [i, i+h] for fully observed contiguous windows."""
    np, pd, _, _ = require_stack()
    out: dict[int, tuple[Any, Any]] = {}
    segs = list(contiguous_segments(sec, price))
    for h in horizons:
        fmax = np.full(len(price), np.nan, dtype=float)
        fmin = np.full(len(price), np.nan, dtype=float)
        w = int(h) + 1
        for a, b in segs:
            if b - a + 1 < w:
                continue
            arr = price[a : b + 1]
            rev = pd.Series(arr[::-1])
            mx = rev.rolling(w, min_periods=w).max().to_numpy()[::-1]
            mn = rev.rolling(w, min_periods=w).min().to_numpy()[::-1]
            fmax[a : b + 1] = mx
            fmin[a : b + 1] = mn
        out[int(h)] = (fmax, fmin)
    return out


def full_horizon_available(sec, price, i: int, h: int) -> bool:
    j = i + int(h)
    return (
        0 <= i < len(sec)
        and j < len(sec)
        and math.isfinite(float(price[i]))
        and math.isfinite(float(price[j]))
        and int(sec[j]) == int(sec[i]) + int(h)
    )


def first_hit_index(price, start_i: int, end_i: int, anchor_price: float, direction: int, target_bps: float):
    np, _, _, _ = require_stack()
    if end_i < start_i:
        return None
    path = price[start_i : end_i + 1]
    if direction > 0:
        cond = (path / anchor_price - 1.0) * 10_000.0 >= float(target_bps) - 1e-12
    else:
        cond = (1.0 - path / anchor_price) * 10_000.0 >= float(target_bps) - 1e-12
    idx = np.flatnonzero(cond)
    return int(start_i + idx[0]) if len(idx) else None


def prehit_mae_bps(price, start_i: int, hit_i: int, anchor_price: float, direction: int) -> float:
    np, _, _, _ = require_stack()
    path = price[start_i : hit_i + 1]
    dr = float(direction) * (path / anchor_price - 1.0) * 10_000.0
    return float(max(0.0, -float(np.nanmin(dr)))) if len(dr) else math.nan


def detection_row(
    definition: str,
    sec,
    price,
    i: int,
    direction: int,
    horizon_sec: int,
    target_bps: float,
    hit_i: int,
    trailing_rv_bps: float,
    terminal_return_bps: float,
    max_favorable_bps: float,
    prehit_mae: float,
) -> dict[str, Any]:
    anchor_second = int(sec[i])
    hit_second = int(sec[hit_i])
    direction_label = "UP" if direction > 0 else "DOWN"
    return {
        "detection_id": stable_id(
            "7I9_ALT_DETECTION", definition, anchor_second, direction, hit_second, round(float(target_bps), 8)
        ),
        "definition": definition,
        "symbol": DEFAULT_SYMBOL,
        "direction": int(direction),
        "direction_label": direction_label,
        "anchor_second": anchor_second,
        "anchor_time": iso_time(anchor_second),
        "anchor_date": datetime.fromtimestamp(anchor_second, tz=timezone.utc).date().isoformat(),
        "anchor_price": float(price[i]),
        "horizon_sec": int(horizon_sec),
        "target_bps": float(target_bps),
        "first_hit_second": hit_second,
        "first_hit_time": iso_time(hit_second),
        "seconds_to_hit": int(hit_second - anchor_second),
        "first_hit_price": float(price[hit_i]),
        "terminal_return_bps": float(terminal_return_bps),
        "max_favorable_bps": float(max_favorable_bps),
        "prehit_mae_bps": float(prehit_mae),
        "trailing_rv15_bps": float(trailing_rv_bps) if math.isfinite(trailing_rv_bps) else math.nan,
        "normalized_target_over_rv": (
            float(target_bps) / float(trailing_rv_bps)
            if math.isfinite(trailing_rv_bps) and trailing_rv_bps > 0
            else math.nan
        ),
        "anchor_contract": f"FIXED_{ANCHOR_CADENCE_SEC}S_GRID_RETROSPECTIVE_AUDIT",
    }


def build_alternate_detections(sec, price, research_start: date, research_end: date, phase7a):
    np, pd, _, _ = require_stack()
    print("      computing forward 5/10/15-minute extrema...")
    ext = future_extreme_arrays(sec, price, FIXED_HORIZONS_SEC)
    print("      computing causal trailing 15-minute realized volatility...")
    rv15 = phase7a.trailing_rv_bps(price, sec, VOL_LOOKBACK_SEC)

    start_epoch = int(datetime.combine(research_start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_epoch_exclusive = int(
        datetime.combine(research_end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp()
    )
    anchor_mask = (
        (sec % ANCHOR_CADENCE_SEC == 0)
        & (sec >= start_epoch)
        & (sec < end_epoch_exclusive)
        & np.isfinite(price)
        & (price > 0)
    )
    anchors = np.flatnonzero(anchor_mask)

    rows: list[dict[str, Any]] = []
    nesting_checks = {
        "fp75_5_not_10": 0,
        "fp75_10_not_15": 0,
        "path_not_fp75_10": 0,
    }

    for k, i in enumerate(anchors):
        ap = float(price[i])
        if k and k % 50000 == 0:
            print(f"      scanned anchors={k:,}/{len(anchors):,} detections={len(rows):,}")

        fp_flags: dict[tuple[int, int], bool] = {}
        fp_hits: dict[tuple[int, int], int | None] = {}
        horizon_available: dict[int, bool] = {}

        for h in FIXED_HORIZONS_SEC:
            fmax, fmin = ext[int(h)]
            available = (
                math.isfinite(float(fmax[i]))
                and math.isfinite(float(fmin[i]))
                and full_horizon_available(sec, price, i, h)
            )
            horizon_available[int(h)] = bool(available)
            if not available:
                continue
            end_i = i + int(h)
            terminal_ret = (float(price[end_i]) / ap - 1.0) * 10_000.0
            up_mfe = (float(fmax[i]) / ap - 1.0) * 10_000.0
            dn_mfe = (1.0 - float(fmin[i]) / ap) * 10_000.0

            # First-passage 75 definitions.
            for direction, mfe in ((+1, up_mfe), (-1, dn_mfe)):
                hit_i = None
                qualifies = mfe + 1e-12 >= FIRST_PASSAGE_TARGET_BPS
                fp_flags[(h, direction)] = bool(qualifies)
                if qualifies:
                    hit_i = first_hit_index(price, i, end_i, ap, direction, FIRST_PASSAGE_TARGET_BPS)
                    fp_hits[(h, direction)] = hit_i
                    if hit_i is not None:
                        mae = prehit_mae_bps(price, i, hit_i, ap, direction)
                        rows.append(
                            detection_row(
                                f"FIRST_PASSAGE_75_{h // 60}M",
                                sec, price, i, direction, h, FIRST_PASSAGE_TARGET_BPS,
                                hit_i, safe_float(rv15[i]), terminal_ret * direction, mfe, mae,
                            )
                        )
                else:
                    fp_hits[(h, direction)] = None

            # Fixed exact-horizon terminal displacement.
            if abs(terminal_ret) + 1e-12 >= FIRST_PASSAGE_TARGET_BPS:
                direction = +1 if terminal_ret > 0 else -1
                hit_i = end_i
                mfe = up_mfe if direction > 0 else dn_mfe
                mae = prehit_mae_bps(price, i, hit_i, ap, direction)
                rows.append(
                    detection_row(
                        f"TERMINAL_75_{h // 60}M",
                        sec, price, i, direction, h, FIRST_PASSAGE_TARGET_BPS,
                        hit_i, safe_float(rv15[i]), abs(terminal_ret), mfe, mae,
                    )
                )

        # Path-efficient 75 within 10m, <=15bp adverse before first hit.
        for direction in (+1, -1):
            hit_i = fp_hits.get((PATH_EFFICIENT_HORIZON_SEC, direction))
            if hit_i is not None:
                mae = prehit_mae_bps(price, i, hit_i, ap, direction)
                if mae <= PATH_EFFICIENT_MAX_PREHIT_MAE_BPS + 1e-12:
                    h = PATH_EFFICIENT_HORIZON_SEC
                    fmax, fmin = ext[h]
                    mfe = (
                        (float(fmax[i]) / ap - 1.0) * 10_000.0
                        if direction > 0
                        else (1.0 - float(fmin[i]) / ap) * 10_000.0
                    )
                    terminal_ret = direction * (float(price[i + h]) / ap - 1.0) * 10_000.0
                    rows.append(
                        detection_row(
                            "PATH_EFFICIENT_75_10M",
                            sec, price, i, direction, h, FIRST_PASSAGE_TARGET_BPS,
                            hit_i, safe_float(rv15[i]), terminal_ret, mfe, mae,
                        )
                    )

        # Volatility-normalized expansion.
        rv = safe_float(rv15[i])
        if math.isfinite(rv) and rv > 0 and full_horizon_available(sec, price, i, VOL_HORIZON_SEC):
            target = max(VOL_TARGET_FLOOR_BPS, VOL_MULTIPLIER * rv)
            fmax, fmin = ext[VOL_HORIZON_SEC]
            up_mfe = (float(fmax[i]) / ap - 1.0) * 10_000.0
            dn_mfe = (1.0 - float(fmin[i]) / ap) * 10_000.0
            for direction, mfe in ((+1, up_mfe), (-1, dn_mfe)):
                if mfe + 1e-12 >= target:
                    end_i = i + VOL_HORIZON_SEC
                    hit_i = first_hit_index(price, i, end_i, ap, direction, target)
                    if hit_i is not None:
                        mae = prehit_mae_bps(price, i, hit_i, ap, direction)
                        terminal_ret = direction * (float(price[end_i]) / ap - 1.0) * 10_000.0
                        rows.append(
                            detection_row(
                                "VOLNORM_3X_RV15_10M",
                                sec, price, i, direction, VOL_HORIZON_SEC, target,
                                hit_i, rv, terminal_ret, mfe, mae,
                            )
                        )

        # 100-bp first passage within 10m.
        h = 600
        fmax, fmin = ext[h]
        if math.isfinite(float(fmax[i])) and math.isfinite(float(fmin[i])) and full_horizon_available(sec, price, i, h):
            end_i = i + h
            up_mfe = (float(fmax[i]) / ap - 1.0) * 10_000.0
            dn_mfe = (1.0 - float(fmin[i]) / ap) * 10_000.0
            for direction, mfe in ((+1, up_mfe), (-1, dn_mfe)):
                if mfe + 1e-12 >= FAST_100_TARGET_BPS:
                    hit_i = first_hit_index(price, i, end_i, ap, direction, FAST_100_TARGET_BPS)
                    if hit_i is not None:
                        mae = prehit_mae_bps(price, i, hit_i, ap, direction)
                        terminal_ret = direction * (float(price[end_i]) / ap - 1.0) * 10_000.0
                        rows.append(
                            detection_row(
                                "FIRST_PASSAGE_100_10M",
                                sec, price, i, direction, h, FAST_100_TARGET_BPS,
                                hit_i, safe_float(rv15[i]), terminal_ret, mfe, mae,
                            )
                        )

        # Anchor-level nesting audit.
        for direction in (+1, -1):
            if (
                horizon_available.get(300, False)
                and horizon_available.get(600, False)
                and fp_flags.get((300, direction), False)
                and not fp_flags.get((600, direction), False)
            ):
                nesting_checks["fp75_5_not_10"] += 1
            if (
                horizon_available.get(600, False)
                and horizon_available.get(900, False)
                and fp_flags.get((600, direction), False)
                and not fp_flags.get((900, direction), False)
            ):
                nesting_checks["fp75_10_not_15"] += 1

    detections = pd.DataFrame(rows)
    if detections.empty:
        detections = pd.DataFrame(columns=[
            "detection_id","definition","symbol","direction","direction_label","anchor_second","anchor_time",
            "anchor_date","anchor_price","horizon_sec","target_bps","first_hit_second","first_hit_time",
            "seconds_to_hit","first_hit_price","terminal_return_bps","max_favorable_bps","prehit_mae_bps",
            "trailing_rv15_bps","normalized_target_over_rv","anchor_contract",
        ])
    else:
        detections = detections.sort_values(
            ["definition", "direction", "anchor_second", "first_hit_second"], kind="mergesort"
        ).reset_index(drop=True)
    return detections, nesting_checks, int(len(anchors))



def _cluster_parent_detection_records(records, family: str):
    """
    Cluster a common parent detection family by first-hit time.

    Important correction vs v1.0.0:
    * clustering is performed ONCE on the broad parent family;
    * stricter nested definitions inherit the resulting impulse_id;
    * the cluster span is bounded against the FIRST hit in the cluster, so
      transitive chains cannot bridge arbitrarily distant accelerations.
    """
    _, pd, _, _ = require_stack()
    rows: list[dict[str, Any]] = []
    if not records:
        return rows

    recs = sorted(records, key=lambda r: (int(r["first_hit_second"]), int(r["anchor_second"])))
    cluster: list[dict[str, Any]] = []
    cluster_first_hit: int | None = None

    def flush(items: list[dict[str, Any]]) -> None:
        if not items:
            return
        anchors = [int(r["anchor_second"]) for r in items]
        hits = [int(r["first_hit_second"]) for r in items]
        # Earliest hit is the physical expansion marker.  If several anchors
        # share that hit, use the latest anchor as the representative because
        # it is closest to the actual acceleration.
        rep = min(items, key=lambda r: (int(r["first_hit_second"]), -int(r["anchor_second"])))
        impulse_id = stable_id(
            "7I9_COMMON_IMPULSE",
            family,
            int(rep["direction"]),
            min(hits),
            max(hits),
            min(anchors),
            max(anchors),
        )
        rows.append({
            "impulse_id": impulse_id,
            "impulse_family": family,
            "direction": int(rep["direction"]),
            "direction_label": str(rep["direction_label"]),
            "cluster_start_anchor_second": min(anchors),
            "cluster_start_anchor_time": iso_time(min(anchors)),
            "cluster_end_anchor_second": max(anchors),
            "cluster_end_anchor_time": iso_time(max(anchors)),
            "representative_anchor_second": int(rep["anchor_second"]),
            "representative_anchor_time": str(rep["anchor_time"]),
            "representative_anchor_price": float(rep["anchor_price"]),
            "representative_hit_second": int(rep["first_hit_second"]),
            "representative_hit_time": str(rep["first_hit_time"]),
            "representative_seconds_to_hit": int(rep["seconds_to_hit"]),
            "parent_detection_count": int(len(items)),
            "parent_hit_span_sec": int(max(hits) - min(hits)),
            "parent_anchor_span_sec": int(max(anchors) - min(anchors)),
        })

    for r in recs:
        hit = int(r["first_hit_second"])
        if not cluster:
            cluster = [r]
            cluster_first_hit = hit
            continue
        assert cluster_first_hit is not None
        if hit - cluster_first_hit <= CLUSTER_MAX_HIT_GAP_SEC:
            cluster.append(r)
        else:
            flush(cluster)
            cluster = [r]
            cluster_first_hit = hit
    flush(cluster)
    return rows


def build_common_impulses_and_events(detections):
    """
    Build one common impulse clustering for nested fixed-75 definitions.

    The fixed-75 parent is the UNION of every available
    FIRST_PASSAGE_75_5M / 10M / 15M anchor-direction detection.

    This matters at data/gap boundaries: a fully observed 5-minute child can be
    scientifically valid even when a complete 15-minute future window does not
    exist.  Requiring a 15-minute parent would incorrectly orphan that child.

    When the same anchor/direction exists at several horizons, its first passage
    to 75 bps is the same physical hit.  We keep one deterministic representative
    (the shortest available horizon) only for common-parent clustering.

    Definitions therefore become labels on common impulses rather than
    independently clustered event sets.

    VOLNORM can use a target below 75 bps, so detections that have no matching
    fixed-75 parent anchor are clustered into a separate VOLNORM_ONLY family.
    """
    _, pd, _, _ = require_stack()
    if detections.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    parent = detections[
        detections["definition"].isin(COMMON_PARENT_SOURCE_DEFINITIONS)
    ].copy()
    if parent.empty:
        raise RuntimeError(
            "No detections found for fixed-75 common parent source definitions"
        )

    # One common-parent row per exact anchor/direction.  The shortest available
    # horizon is kept only as a deterministic representative; all child labels
    # still remain in `detections`.
    parent = (
        parent.sort_values(
            ["anchor_second", "direction", "horizon_sec", "first_hit_second"],
            kind="mergesort",
        )
        .drop_duplicates(["anchor_second", "direction"], keep="first")
        .reset_index(drop=True)
    )

    impulse_rows: list[dict[str, Any]] = []
    anchor_to_impulse: dict[tuple[int, int], str] = {}

    for direction, g in parent.groupby("direction", sort=False):
        clusters = _cluster_parent_detection_records(g.to_dict("records"), "FIXED_75_PARENT")
        impulse_rows.extend(clusters)

    # Assign every parent raw anchor to the unique common impulse containing it.
    # Reconstruct membership from bounded hit-time clusters deterministically.
    for direction, g in parent.groupby("direction", sort=False):
        recs = sorted(g.to_dict("records"), key=lambda r: (int(r["first_hit_second"]), int(r["anchor_second"])))
        impulse_subset = sorted(
            [r for r in impulse_rows if int(r["direction"]) == int(direction) and r["impulse_family"] == "FIXED_75_PARENT"],
            key=lambda r: int(r["representative_hit_second"]),
        )
        # Because clusters are ordered/non-overlapping in hit time, map using
        # the cluster's representative/maximum hit span.
        for r in recs:
            hit = int(r["first_hit_second"])
            candidates = [
                imp for imp in impulse_subset
                if int(imp["representative_hit_second"]) <= hit
                and hit <= int(imp["representative_hit_second"]) + CLUSTER_MAX_HIT_GAP_SEC
            ]
            if not candidates:
                # Defensive nearest-hit fallback inside the bounded family.
                candidates = sorted(
                    impulse_subset,
                    key=lambda imp: abs(hit - int(imp["representative_hit_second"]))
                )[:1]
            if not candidates:
                raise RuntimeError("Could not map common-parent detection to an impulse")
            anchor_to_impulse[(int(r["anchor_second"]), int(direction))] = str(candidates[0]["impulse_id"])

    # Supplemental volatility-normalized detections that are not represented by
    # a fixed-75 parent anchor get their own common clustering.
    vol = detections[detections["definition"] == "VOLNORM_3X_RV15_10M"].copy()
    vol_unmapped = []
    for r in vol.to_dict("records"):
        key = (int(r["anchor_second"]), int(r["direction"]))
        if key not in anchor_to_impulse:
            vol_unmapped.append(r)
    for direction in (-1, +1):
        recs = [r for r in vol_unmapped if int(r["direction"]) == direction]
        clusters = _cluster_parent_detection_records(recs, "VOLNORM_ONLY")
        impulse_rows.extend(clusters)

    # Map supplemental VOLNORM anchors to their supplemental impulses.
    vol_impulses = [
        r for r in impulse_rows if r["impulse_family"] == "VOLNORM_ONLY"
    ]
    for direction in (-1, +1):
        recs = sorted(
            [r for r in vol_unmapped if int(r["direction"]) == direction],
            key=lambda r: (int(r["first_hit_second"]), int(r["anchor_second"]))
        )
        imps = sorted(
            [r for r in vol_impulses if int(r["direction"]) == direction],
            key=lambda r: int(r["representative_hit_second"])
        )
        for r in recs:
            hit = int(r["first_hit_second"])
            candidates = [
                imp for imp in imps
                if int(imp["representative_hit_second"]) <= hit
                and hit <= int(imp["representative_hit_second"]) + CLUSTER_MAX_HIT_GAP_SEC
            ]
            if not candidates:
                candidates = sorted(imps, key=lambda imp: abs(hit - int(imp["representative_hit_second"])))[:1]
            if not candidates:
                raise RuntimeError("Could not map VOLNORM-only detection to an impulse")
            anchor_to_impulse[(int(r["anchor_second"]), int(direction))] = str(candidates[0]["impulse_id"])

    impulses = pd.DataFrame(impulse_rows).sort_values(
        ["representative_hit_second", "direction", "impulse_family"], kind="mergesort"
    ).reset_index(drop=True)

    # Every raw detection now becomes a LABEL on the common impulse determined
    # by its exact anchor/direction.  This is the core v1.0.1 correction.
    labeled = detections.copy()
    labeled["impulse_id"] = [
        anchor_to_impulse.get((int(a), int(d)))
        for a, d in zip(labeled["anchor_second"], labeled["direction"])
    ]
    missing_nested = (
        labeled["definition"].isin(COMMON_PARENT_DEFINITIONS)
        & labeled["impulse_id"].isna()
    )
    if missing_nested.any():
        bad = labeled.loc[missing_nested, ["definition","anchor_second","direction"]].head(20)
        raise RuntimeError(
            "Nested alternate detections could not inherit a common impulse ID:\n"
            + bad.to_string(index=False)
        )

    # A VOLNORM detection may share a fixed-75 anchor; otherwise it was assigned
    # to VOLNORM_ONLY above.
    missing_any = labeled["impulse_id"].isna()
    if missing_any.any():
        bad = labeled.loc[missing_any, ["definition","anchor_second","direction"]].head(20)
        raise RuntimeError(
            "Alternate detections missing common impulse IDs:\n" + bad.to_string(index=False)
        )

    # Positive first-passage labels are logically nested even when a longer
    # future window is censored. Once 75 bps has actually been observed within
    # 5m, for example, the statement "75 within 10m" is already known TRUE;
    # no unseen future path is needed.  Propagate positive labels upward while
    # leaving negatives conservative/full-horizon-only.
    labeled["label_provenance"] = "RAW_DETECTION"
    implied: list[dict[str, Any]] = []
    horizon_for = {
        "FIRST_PASSAGE_75_5M": 300,
        "FIRST_PASSAGE_75_10M": 600,
        "FIRST_PASSAGE_75_15M": 900,
    }
    implication_map = {
        "FIRST_PASSAGE_75_5M": ("FIRST_PASSAGE_75_10M", "FIRST_PASSAGE_75_15M"),
        "FIRST_PASSAGE_75_10M": ("FIRST_PASSAGE_75_15M",),
        "TERMINAL_75_5M": ("FIRST_PASSAGE_75_5M", "FIRST_PASSAGE_75_10M", "FIRST_PASSAGE_75_15M"),
        "TERMINAL_75_10M": ("FIRST_PASSAGE_75_10M", "FIRST_PASSAGE_75_15M"),
        "TERMINAL_75_15M": ("FIRST_PASSAGE_75_15M",),
        "PATH_EFFICIENT_75_10M": ("FIRST_PASSAGE_75_10M", "FIRST_PASSAGE_75_15M"),
        "FIRST_PASSAGE_100_10M": ("FIRST_PASSAGE_75_10M", "FIRST_PASSAGE_75_15M"),
    }
    existing = set(
        labeled[["definition", "anchor_second", "direction"]]
        .itertuples(index=False, name=None)
    )
    for r in labeled.to_dict("records"):
        for implied_def in implication_map.get(str(r["definition"]), ()):
            key = (implied_def, int(r["anchor_second"]), int(r["direction"]))
            if key in existing:
                continue
            rr = dict(r)
            rr["definition"] = implied_def
            rr["horizon_sec"] = int(horizon_for[implied_def])
            rr["target_bps"] = 75.0
            rr["terminal_return_bps"] = math.nan
            rr["label_provenance"] = f"INHERITED_POSITIVE_FROM_{r['definition']}"
            implied.append(rr)
            existing.add(key)
    if implied:
        labeled = pd.concat([labeled, pd.DataFrame(implied)], ignore_index=True, sort=False)

    event_rows: list[dict[str, Any]] = []
    for (impulse_id, definition), g in labeled.groupby(["impulse_id","definition"], sort=False):
        recs = g.to_dict("records")
        # Representative for this definition remains definition-specific, while
        # the impulse ID itself is common across nested definitions.
        rep = min(recs, key=lambda r: (int(r["first_hit_second"]), -int(r["anchor_second"])))
        imp = impulses.loc[impulses["impulse_id"] == impulse_id].iloc[0]
        event_rows.append({
            "alternate_event_id": stable_id("7I9_ALT_EVENT_LABEL", impulse_id, definition),
            "impulse_id": str(impulse_id),
            "impulse_family": str(imp["impulse_family"]),
            "definition": str(definition),
            "direction": int(rep["direction"]),
            "direction_label": str(rep["direction_label"]),
            "cluster_start_anchor_second": int(imp["cluster_start_anchor_second"]),
            "cluster_start_anchor_time": str(imp["cluster_start_anchor_time"]),
            "cluster_end_anchor_second": int(imp["cluster_end_anchor_second"]),
            "cluster_end_anchor_time": str(imp["cluster_end_anchor_time"]),
            "representative_anchor_second": int(rep["anchor_second"]),
            "representative_anchor_time": str(rep["anchor_time"]),
            "representative_anchor_price": float(rep["anchor_price"]),
            "representative_hit_second": int(rep["first_hit_second"]),
            "representative_hit_time": str(rep["first_hit_time"]),
            "representative_seconds_to_hit": int(rep["seconds_to_hit"]),
            "target_bps": float(rep["target_bps"]),
            "horizon_sec": int(rep["horizon_sec"]),
            "detection_count": int(len(recs)),
            "label_provenance": "|".join(sorted(set(str(r.get("label_provenance", "RAW_DETECTION")) for r in recs))),
            "max_favorable_bps": float(max(safe_float(r["max_favorable_bps"]) for r in recs)),
            "min_prehit_mae_bps": float(min(safe_float(r["prehit_mae_bps"]) for r in recs)),
            "median_prehit_mae_bps": float(pd.Series([safe_float(r["prehit_mae_bps"]) for r in recs]).median()),
            "median_trailing_rv15_bps": float(pd.Series([safe_float(r["trailing_rv15_bps"]) for r in recs]).median()),
            "cluster_span_sec": int(imp["parent_anchor_span_sec"]),
        })

    events = pd.DataFrame(event_rows).sort_values(
        ["representative_hit_second","definition","direction"], kind="mergesort"
    ).reset_index(drop=True)

    nested_rows = []
    label_sets = {
        definition: set(events.loc[events["definition"] == definition, "impulse_id"].astype(str))
        for definition in DEFINITION_ORDER
    }
    checks = (
        ("FIRST_PASSAGE_75_5M", "FIRST_PASSAGE_75_10M"),
        ("FIRST_PASSAGE_75_10M", "FIRST_PASSAGE_75_15M"),
        ("PATH_EFFICIENT_75_10M", "FIRST_PASSAGE_75_10M"),
        ("FIRST_PASSAGE_100_10M", "FIRST_PASSAGE_75_10M"),
        ("TERMINAL_75_5M", "FIRST_PASSAGE_75_5M"),
        ("TERMINAL_75_10M", "FIRST_PASSAGE_75_10M"),
        ("TERMINAL_75_15M", "FIRST_PASSAGE_75_15M"),
    )
    for child, parent_name in checks:
        child_set = label_sets.get(child, set())
        parent_set = label_sets.get(parent_name, set())
        violations = child_set - parent_set
        nested_rows.append({
            "child_definition": child,
            "parent_definition": parent_name,
            "child_impulses": len(child_set),
            "parent_impulses": len(parent_set),
            "subset_violation_count": len(violations),
            "status": "PASS" if not violations else "FAIL",
        })
    nested_consistency = pd.DataFrame(nested_rows)

    return impulses, events, nested_consistency


def cluster_detections(detections):
    """Compatibility wrapper used only by the self-test."""
    _, events, _ = build_common_impulses_and_events(detections)
    return events

def load_frozen_tables(phase7a_root: Path, phase7i1_root: Path, phase7i6_root: Path):
    _, _, _, pq = require_parquet_stack()
    paths = {
        "physical": phase7a_root / "physical_events.parquet",
        "episodes": phase7i1_root / "episodes.parquet",
        "candidates": phase7i1_root / "candidates.parquet",
        "geometry": phase7i1_root / "geometry_audit.parquet",
        "analysis": phase7i6_root / "participant_interaction_analysis.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError("Missing frozen input datasets:\n  " + "\n  ".join(missing))
    return {k: pq.read_table(v).to_pandas() for k, v in paths.items()}


def _interval_end(rec: Mapping[str, Any], preferred: Sequence[str], fallback: int) -> int:
    for c in preferred:
        v = safe_float(rec.get(c))
        if math.isfinite(v):
            return int(v)
    return int(fallback)


def map_alternate_events(events, physical, episodes, candidates):
    _, pd, _, _ = require_stack()
    if events.empty:
        return events.copy()

    c_by_episode = {
        str(r["episode_id"]): r
        for r in candidates.to_dict("records")
    }

    def build_interval_index(frame, start_key: str, end_keys: Sequence[str]):
        idx = {}
        for d, g in frame.groupby(pd.to_numeric(frame["direction"], errors="coerce")):
            recs = g.sort_values(start_key, kind="mergesort").to_dict("records")
            starts = [int(r[start_key]) for r in recs]
            ends = [_interval_end(r, end_keys, int(r[start_key])) for r in recs]
            idx[int(d)] = (starts, ends, recs)
        return idx

    physical_idx = build_interval_index(
        physical,
        "onset_second",
        ("reversal_confirm_second", "extreme_second"),
    )
    episode_idx = build_interval_index(
        episodes,
        "origin_second",
        ("close_second", "extreme_second"),
    )

    def containing(index, direction: int, t: int):
        pack = index.get(int(direction))
        if pack is None:
            return None
        starts, ends, recs = pack
        j = bisect_right(starts, int(t)) - 1
        if j < 0:
            return None
        # Directional-change legs/episodes are sequential, but inspect a few
        # predecessors defensively in case a censored boundary creates overlap.
        for k in range(j, max(-1, j - 4), -1):
            if starts[k] <= int(t) <= ends[k]:
                return recs[k]
        return None

    mapped_rows = []
    for ev in events.to_dict("records"):
        d = int(ev["direction"])
        hit = int(ev["representative_hit_second"])
        anchor = int(ev["representative_anchor_second"])

        ph = containing(physical_idx, d, hit)
        ep = containing(episode_idx, d, hit)

        cand = c_by_episode.get(str(ep["episode_id"])) if ep is not None else None
        if cand is not None and int(cand["candidate_second"]) > hit:
            cand = None

        row = dict(ev)
        row.update({
            "mapped_phase7a_event": ph is not None,
            "phase7a_event_id": str(ph["event_id"]) if ph is not None else None,
            "phase7a_time_to_75_sec": safe_float(ph.get("time_to_75_sec")) if ph is not None else math.nan,
            "phase7a_time_to_100_sec": safe_float(ph.get("time_to_100_sec")) if ph is not None else math.nan,
            "covered_by_current_7a_75_30m": bool(
                ph is not None
                and math.isfinite(safe_float(ph.get("time_to_75_sec")))
                and safe_float(ph.get("time_to_75_sec")) <= 1800
            ),
            "covered_by_current_7a_100_30m": bool(
                ph is not None
                and math.isfinite(safe_float(ph.get("time_to_100_sec")))
                and safe_float(ph.get("time_to_100_sec")) <= 1800
            ),
            "mapped_causal_episode": ep is not None,
            "causal_episode_id": str(ep["episode_id"]) if ep is not None else None,
            "covered_by_causal_35_candidate": cand is not None,
            "candidate_id": str(cand["candidate_id"]) if cand is not None else None,
            "candidate_second": int(cand["candidate_second"]) if cand is not None else math.nan,
            "candidate_lead_to_alt_hit_sec": int(hit - int(cand["candidate_second"])) if cand is not None else math.nan,
            "alternate_anchor_minus_candidate_sec": int(anchor - int(cand["candidate_second"])) if cand is not None else math.nan,
        })
        mapped_rows.append(row)

    return pd.DataFrame(mapped_rows)

def build_candidate_membership(candidates, geometry, physical, mapped_events):
    _, pd, _, _ = require_stack()
    out = candidates[["candidate_id","episode_id","candidate_second","candidate_date","direction","direction_label"]].copy()

    phys = physical[["event_id","time_to_75_sec","time_to_100_sec"]].copy()
    phys["event_id"] = phys["event_id"].astype(str)
    geom = geometry[["candidate_id","phase7a_event_id"]].copy()
    geom["candidate_id"] = geom["candidate_id"].astype(str)
    geom["phase7a_event_id"] = geom["phase7a_event_id"].astype("string")
    out["candidate_id"] = out["candidate_id"].astype(str)
    out = out.merge(geom, on="candidate_id", how="left")
    out = out.merge(phys, left_on="phase7a_event_id", right_on="event_id", how="left")
    out[CURRENT_PRIMARY_DEFINITION] = (
        pd.to_numeric(out["time_to_75_sec"], errors="coerce").notna()
        & (pd.to_numeric(out["time_to_75_sec"], errors="coerce") <= 1800)
    )

    for definition in DEFINITION_ORDER:
        ids = set(
            mapped_events.loc[
                (mapped_events["definition"] == definition)
                & mapped_events["covered_by_causal_35_candidate"].fillna(False).astype(bool),
                "candidate_id",
            ].dropna().astype(str)
        )
        out[definition] = out["candidate_id"].isin(ids)

    keep = [
        "candidate_id","episode_id","candidate_second","candidate_date","direction","direction_label",
        CURRENT_PRIMARY_DEFINITION,*DEFINITION_ORDER,
    ]
    return out[keep].copy()


def definition_summary(mapped_events, physical, detections):
    _, pd, _, _ = require_stack()
    rows = []
    current75 = physical[
        pd.to_numeric(physical["time_to_75_sec"], errors="coerce").notna()
        & (pd.to_numeric(physical["time_to_75_sec"], errors="coerce") <= 1800)
    ]
    current100 = physical[
        pd.to_numeric(physical["time_to_100_sec"], errors="coerce").notna()
        & (pd.to_numeric(physical["time_to_100_sec"], errors="coerce") <= 1800)
    ]
    rows.append({
        "definition": CURRENT_PRIMARY_DEFINITION,
        "raw_detections": math.nan,
        "clustered_events": int(len(current75)),
        "up_events": int((pd.to_numeric(current75["direction"], errors="coerce") > 0).sum()),
        "down_events": int((pd.to_numeric(current75["direction"], errors="coerce") < 0).sum()),
        "median_seconds_to_hit": float(pd.to_numeric(current75["time_to_75_sec"], errors="coerce").median()),
        "median_prehit_mae_bps": math.nan,
        "covered_by_current_7a_75_30m_rate": 1.0,
        "covered_by_causal_candidate_rate": math.nan,
    })
    rows.append({
        "definition": "CURRENT_7A_100_WITHIN_30M",
        "raw_detections": math.nan,
        "clustered_events": int(len(current100)),
        "up_events": int((pd.to_numeric(current100["direction"], errors="coerce") > 0).sum()),
        "down_events": int((pd.to_numeric(current100["direction"], errors="coerce") < 0).sum()),
        "median_seconds_to_hit": float(pd.to_numeric(current100["time_to_100_sec"], errors="coerce").median()),
        "median_prehit_mae_bps": math.nan,
        "covered_by_current_7a_75_30m_rate": 1.0,
        "covered_by_causal_candidate_rate": math.nan,
    })
    for definition in DEFINITION_ORDER:
        e = mapped_events[mapped_events["definition"] == definition]
        d = detections[detections["definition"] == definition]
        rows.append({
            "definition": definition,
            "raw_detections": int(len(d)),
            "clustered_events": int(len(e)),
            "up_events": int((pd.to_numeric(e["direction"], errors="coerce") > 0).sum()) if len(e) else 0,
            "down_events": int((pd.to_numeric(e["direction"], errors="coerce") < 0).sum()) if len(e) else 0,
            "median_seconds_to_hit": float(pd.to_numeric(e["representative_seconds_to_hit"], errors="coerce").median()) if len(e) else math.nan,
            "median_prehit_mae_bps": float(pd.to_numeric(e["median_prehit_mae_bps"], errors="coerce").median()) if len(e) else math.nan,
            "covered_by_current_7a_75_30m_rate": float(e["covered_by_current_7a_75_30m"].fillna(False).mean()) if len(e) else math.nan,
            "covered_by_causal_candidate_rate": float(e["covered_by_causal_35_candidate"].fillna(False).mean()) if len(e) else math.nan,
        })
    return pd.DataFrame(rows)


def coverage_report(mapped_events):
    _, pd, _, _ = require_stack()
    rows = []
    for definition, e in mapped_events.groupby("definition", sort=False):
        rows.append({
            "definition": definition,
            "events": int(len(e)),
            "mapped_any_phase7a_leg": int(e["mapped_phase7a_event"].fillna(False).sum()),
            "mapped_any_phase7a_leg_rate": float(e["mapped_phase7a_event"].fillna(False).mean()),
            "covered_current_7a_75_30m": int(e["covered_by_current_7a_75_30m"].fillna(False).sum()),
            "covered_current_7a_75_30m_rate": float(e["covered_by_current_7a_75_30m"].fillna(False).mean()),
            "covered_causal_35_candidate": int(e["covered_by_causal_35_candidate"].fillna(False).sum()),
            "covered_causal_35_candidate_rate": float(e["covered_by_causal_35_candidate"].fillna(False).mean()),
            "missed_by_both": int(
                (
                    ~e["covered_by_current_7a_75_30m"].fillna(False).astype(bool)
                    & ~e["covered_by_causal_35_candidate"].fillna(False).astype(bool)
                ).sum()
            ),
        })
    return pd.DataFrame(rows)


def candidate_membership_rates(membership):
    _, pd, _, _ = require_stack()
    rows = []
    n = len(membership)
    for definition in (CURRENT_PRIMARY_DEFINITION, *DEFINITION_ORDER):
        s = membership[definition].fillna(False).astype(bool)
        rows.append({
            "definition": definition,
            "candidates": int(n),
            "member_candidates": int(s.sum()),
            "membership_rate": float(s.mean()) if n else math.nan,
            "up_membership_rate": float(s[pd.to_numeric(membership["direction"], errors="coerce") > 0].mean()),
            "down_membership_rate": float(s[pd.to_numeric(membership["direction"], errors="coerce") < 0].mean()),
        })
    return pd.DataFrame(rows)


def overlap_matrix(membership):
    _, pd, _, _ = require_stack()
    defs = [CURRENT_PRIMARY_DEFINITION, *DEFINITION_ORDER]
    rows = []
    for a in defs:
        A = membership[a].fillna(False).astype(bool)
        for b in defs:
            B = membership[b].fillna(False).astype(bool)
            inter = int((A & B).sum())
            union = int((A | B).sum())
            rows.append({
                "definition_a": a,
                "definition_b": b,
                "intersection_candidates": inter,
                "union_candidates": union,
                "jaccard": float(inter / union) if union else math.nan,
                "a_covered_by_b_rate": float(inter / int(A.sum())) if int(A.sum()) else math.nan,
            })
    return pd.DataFrame(rows)


def load_book_state(analysis):
    _, pd, _, _ = require_stack()
    a = analysis.copy()
    horizon_col = "observation_horizon_sec" if "observation_horizon_sec" in a.columns else "decision_horizon_sec"
    required = {"candidate_id", horizon_col, "book_quartile", "book_failure_score", "direction"}
    missing = required - set(a.columns)
    if missing:
        raise RuntimeError(f"Phase-7I.6 analysis missing required book-state fields: {sorted(missing)}")
    a = a[pd.to_numeric(a[horizon_col], errors="coerce") == BOOK_DECISION_HORIZON_SEC].copy()
    a["candidate_id"] = a["candidate_id"].astype(str)
    return a[["candidate_id","book_quartile","book_failure_score","direction"]].copy()


def book_failure_robustness(membership, book):
    _, pd, _, _ = require_stack()
    d = membership.merge(book, on="candidate_id", how="inner", suffixes=("","_book"))
    rows = []
    dir_rows = []
    for definition in (CURRENT_PRIMARY_DEFINITION, *DEFINITION_ORDER):
        target = d[definition].fillna(False).astype(bool)
        q = pd.to_numeric(d["book_quartile"], errors="coerce")
        q1 = q == 1
        q4 = q == 4
        r1 = float(target[q1].mean()) if q1.sum() else math.nan
        r4 = float(target[q4].mean()) if q4.sum() else math.nan
        rows.append({
            "definition": definition,
            "book_q1_n": int(q1.sum()),
            "book_q4_n": int(q4.sum()),
            "membership_rate_book_q1": r1,
            "membership_rate_book_q4": r4,
            "q4_minus_q1_pp": (r4 - r1) * 100.0 if math.isfinite(r1) and math.isfinite(r4) else math.nan,
            "rate_ratio_q4_vs_q1": r4 / r1 if math.isfinite(r1) and r1 > 0 and math.isfinite(r4) else math.nan,
        })
        for direction in (-1, +1):
            dm = pd.to_numeric(d["direction"], errors="coerce") == direction
            q1d = q1 & dm
            q4d = q4 & dm
            a = float(target[q1d].mean()) if q1d.sum() else math.nan
            b = float(target[q4d].mean()) if q4d.sum() else math.nan
            dir_rows.append({
                "definition": definition,
                "direction": direction,
                "direction_label": "UP" if direction > 0 else "DOWN",
                "book_q1_n": int(q1d.sum()),
                "book_q4_n": int(q4d.sum()),
                "membership_rate_book_q1": a,
                "membership_rate_book_q4": b,
                "q4_minus_q1_pp": (b - a) * 100.0 if math.isfinite(a) and math.isfinite(b) else math.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(dir_rows)


def missed_fast_events(mapped_events):
    _, pd, _, _ = require_stack()
    focus = {
        "FIRST_PASSAGE_75_5M",
        "FIRST_PASSAGE_75_10M",
        "PATH_EFFICIENT_75_10M",
        "FIRST_PASSAGE_100_10M",
    }
    m = mapped_events[
        mapped_events["definition"].isin(focus)
        & ~mapped_events["covered_by_current_7a_75_30m"].fillna(False).astype(bool)
    ].copy()
    if m.empty:
        return m
    m["economic_priority_score"] = (
        pd.to_numeric(m["target_bps"], errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(m["max_favorable_bps"], errors="coerce").fillna(0)
        - 0.50 * pd.to_numeric(m["median_prehit_mae_bps"], errors="coerce").fillna(0)
        - 0.01 * pd.to_numeric(m["representative_seconds_to_hit"], errors="coerce").fillna(0)
    )
    cols = [
        "alternate_event_id","definition","direction_label","representative_anchor_time",
        "representative_hit_time","representative_seconds_to_hit","target_bps","max_favorable_bps",
        "median_prehit_mae_bps","covered_by_causal_35_candidate","candidate_id",
        "mapped_phase7a_event","phase7a_event_id","economic_priority_score",
    ]
    return m.sort_values("economic_priority_score", ascending=False, kind="mergesort")[cols].head(250)



def physical_leg_impulse_multiplicity(mapped_impulses, mapped_events):
    _, pd, _, _ = require_stack()
    fixed = mapped_impulses[
        (mapped_impulses["impulse_family"] == "FIXED_75_PARENT")
        & mapped_impulses["mapped_phase7a_event"].fillna(False).astype(bool)
    ].copy()
    if fixed.empty:
        return pd.DataFrame()
    label_map = (
        mapped_events.groupby("impulse_id")["definition"]
        .apply(lambda x: "|".join(sorted(set(map(str, x)))))
        .to_dict()
    )
    fixed["definitions"] = fixed["impulse_id"].astype(str).map(label_map).fillna("")
    rows = []
    for event_id, g in fixed.groupby("phase7a_event_id", sort=False):
        g = g.sort_values("representative_hit_second", kind="mergesort")
        rows.append({
            "phase7a_event_id": str(event_id),
            "direction_label": str(g.iloc[0]["direction_label"]),
            "distinct_fixed75_impulses": int(g["impulse_id"].nunique()),
            "first_impulse_hit_time": str(g.iloc[0]["representative_hit_time"]),
            "last_impulse_hit_time": str(g.iloc[-1]["representative_hit_time"]),
            "first_to_last_impulse_sec": int(
                int(g.iloc[-1]["representative_hit_second"])
                - int(g.iloc[0]["representative_hit_second"])
            ),
            "causal_candidate_covered_impulses": int(g["covered_by_causal_35_candidate"].fillna(False).sum()),
            "impulse_ids": "|".join(g["impulse_id"].astype(str)),
            "definitions": " || ".join(g["definitions"].astype(str)),
        })
    return pd.DataFrame(rows).sort_values(
        ["distinct_fixed75_impulses","first_to_last_impulse_sec"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def reacceleration_impulses(mapped_impulses, mapped_events, multiplicity):
    _, pd, _, _ = require_stack()
    if multiplicity.empty:
        return pd.DataFrame()
    multi_ids = set(
        multiplicity.loc[multiplicity["distinct_fixed75_impulses"] > 1, "phase7a_event_id"].astype(str)
    )
    if not multi_ids:
        return pd.DataFrame()
    fixed = mapped_impulses[
        (mapped_impulses["impulse_family"] == "FIXED_75_PARENT")
        & mapped_impulses["phase7a_event_id"].astype(str).isin(multi_ids)
    ].copy()
    label_map = (
        mapped_events.groupby("impulse_id")["definition"]
        .apply(lambda x: "|".join(sorted(set(map(str, x)))))
        .to_dict()
    )
    fixed["definitions"] = fixed["impulse_id"].astype(str).map(label_map).fillna("")
    rows = []
    for event_id, g in fixed.groupby("phase7a_event_id", sort=False):
        g = g.sort_values("representative_hit_second", kind="mergesort").reset_index(drop=True)
        prev_hit = None
        for j, r in g.iterrows():
            hit = int(r["representative_hit_second"])
            rows.append({
                "phase7a_event_id": str(event_id),
                "direction_label": str(r["direction_label"]),
                "impulse_ordinal_in_physical_leg": int(j + 1),
                "impulse_id": str(r["impulse_id"]),
                "representative_anchor_time": str(r["representative_anchor_time"]),
                "representative_hit_time": str(r["representative_hit_time"]),
                "gap_from_previous_impulse_hit_sec": (hit - prev_hit) if prev_hit is not None else math.nan,
                "covered_by_current_7a_75_30m": bool(r["covered_by_current_7a_75_30m"]),
                "covered_by_causal_35_candidate": bool(r["covered_by_causal_35_candidate"]),
                "candidate_id": r.get("candidate_id"),
                "definitions": str(r["definitions"]),
            })
            prev_hit = hit
    return pd.DataFrame(rows)


def corrected_missed_fast_impulses(mapped_impulses, mapped_events):
    _, pd, _, _ = require_stack()
    focus = {
        "FIRST_PASSAGE_75_5M",
        "FIRST_PASSAGE_75_10M",
        "PATH_EFFICIENT_75_10M",
        "FIRST_PASSAGE_100_10M",
    }
    labels = mapped_events[mapped_events["definition"].isin(focus)].copy()
    if labels.empty:
        return pd.DataFrame()
    agg = labels.groupby("impulse_id").agg(
        qualifying_definitions=("definition", lambda x: "|".join(sorted(set(map(str, x))))),
        best_target_bps=("target_bps", "max"),
        fastest_seconds_to_hit=("representative_seconds_to_hit", "min"),
        best_max_favorable_bps=("max_favorable_bps", "max"),
        best_median_prehit_mae_bps=("median_prehit_mae_bps", "min"),
    ).reset_index()
    m = mapped_impulses.merge(agg, on="impulse_id", how="inner")
    m = m[~m["covered_by_current_7a_75_30m"].fillna(False).astype(bool)].copy()
    if m.empty:
        return m
    m["economic_priority_score"] = (
        pd.to_numeric(m["best_target_bps"], errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(m["best_max_favorable_bps"], errors="coerce").fillna(0)
        - 0.50 * pd.to_numeric(m["best_median_prehit_mae_bps"], errors="coerce").fillna(0)
        - 0.01 * pd.to_numeric(m["fastest_seconds_to_hit"], errors="coerce").fillna(0)
    )
    cols = [
        "impulse_id","impulse_family","direction_label","representative_anchor_time",
        "representative_hit_time","qualifying_definitions","best_target_bps",
        "fastest_seconds_to_hit","best_max_favorable_bps","best_median_prehit_mae_bps",
        "mapped_phase7a_event","phase7a_event_id","covered_by_causal_35_candidate",
        "candidate_id","economic_priority_score",
    ]
    return m.sort_values("economic_priority_score", ascending=False, kind="mergesort")[cols].head(250)

def build_invariants(detections, events, membership, nesting_checks, anchor_count, book, impulses=None, nested_consistency=None):
    _, pd, _, _ = require_stack()
    rows = []

    def add(name: str, ok: bool, observed: Any, expected: Any) -> None:
        rows.append({
            "invariant": name,
            "status": "PASS" if bool(ok) else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    add("anchor_grid_nonempty", anchor_count > 0, anchor_count, ">0")
    add("alternate_detections_nonempty", len(detections) > 0, len(detections), ">0")
    add("alternate_events_nonempty", len(events) > 0, len(events), ">0")
    add("detection_ids_unique", not detections["detection_id"].duplicated().any(), int(detections["detection_id"].duplicated().sum()), 0)
    add("alternate_event_ids_unique", not events["alternate_event_id"].duplicated().any(), int(events["alternate_event_id"].duplicated().sum()), 0)
    add("candidate_membership_one_row_per_candidate", not membership["candidate_id"].duplicated().any(), int(membership["candidate_id"].duplicated().sum()), 0)
    add("first_passage_5m_nested_in_10m_at_anchor", nesting_checks["fp75_5_not_10"] == 0, nesting_checks["fp75_5_not_10"], 0)
    add("first_passage_10m_nested_in_15m_at_anchor", nesting_checks["fp75_10_not_15"] == 0, nesting_checks["fp75_10_not_15"], 0)
    add(
        "path_efficient_detections_are_75_10m_subset",
        set(detections.loc[detections["definition"] == "PATH_EFFICIENT_75_10M", ["anchor_second","direction"]].itertuples(index=False, name=None))
        <= set(detections.loc[detections["definition"] == "FIRST_PASSAGE_75_10M", ["anchor_second","direction"]].itertuples(index=False, name=None)),
        "subset_check",
        True,
    )
    add(
        "book_state_one_3s_row_per_candidate",
        not book["candidate_id"].duplicated().any(),
        int(book["candidate_id"].duplicated().sum()),
        0,
    )
    add(
        "candidate_membership_definitions_boolean",
        all(str(membership[c].dtype) in ("bool", "boolean") for c in (CURRENT_PRIMARY_DEFINITION, *DEFINITION_ORDER)),
        {c: str(membership[c].dtype) for c in (CURRENT_PRIMARY_DEFINITION, *DEFINITION_ORDER)},
        "bool",
    )
    if impulses is not None and len(impulses):
        add(
            "common_impulse_ids_unique",
            not impulses["impulse_id"].duplicated().any(),
            int(impulses["impulse_id"].duplicated().sum()),
            0,
        )
    if nested_consistency is not None and len(nested_consistency):
        violations = int(pd.to_numeric(nested_consistency["subset_violation_count"], errors="coerce").fillna(0).sum())
        add(
            "nested_definitions_inherit_common_impulse_ids",
            violations == 0,
            violations,
            0,
        )
    return pd.DataFrame(rows)


def findings_text(summary: Mapping[str, Any], def_summary, coverage, book_robust, missed) -> str:
    def row(df, name):
        x = df[df["definition"] == name]
        return x.iloc[0] if len(x) else None

    lines = [
        "# Phase 7I.9 — Large-move ontology robustness audit",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{summary['status']}**",
        f"- Anchor cadence: **{ANCHOR_CADENCE_SEC}s**",
        f"- Alternate raw detections: **{summary['alternate_detection_count']:,}**",
        f"- Common impulses: **{summary['common_impulse_count']:,}**",
        f"- Alternate event labels: **{summary['alternate_event_label_count']:,}**",
        f"- Frozen causal candidates: **{summary['candidate_count']:,}**",
        "",
        "## Definition coverage",
        "",
    ]
    for name in ("FIRST_PASSAGE_75_5M","FIRST_PASSAGE_75_10M","PATH_EFFICIENT_75_10M","FIRST_PASSAGE_100_10M","VOLNORM_3X_RV15_10M"):
        r = row(coverage, name)
        if r is not None:
            lines.append(
                f"- {name}: events={int(r['events']):,}; "
                f"current-7A-75/30m coverage={100*safe_float(r['covered_current_7a_75_30m_rate']):.1f}%; "
                f"causal-candidate coverage={100*safe_float(r['covered_causal_35_candidate_rate']):.1f}%."
            )
    lines += [
        "",
        "## Book-failure robustness on frozen causal candidates",
        "",
        "This section does not replay book data.  It asks whether the already-frozen",
        "3-second book-failure state is associated with membership in each alternate",
        "large-move definition.",
        "",
    ]
    for name in ("FIRST_PASSAGE_75_5M","FIRST_PASSAGE_75_10M","PATH_EFFICIENT_75_10M","FIRST_PASSAGE_100_10M"):
        r = row(book_robust, name)
        if r is not None:
            lines.append(
                f"- {name}: Q4-Q1 membership difference = {safe_float(r['q4_minus_q1_pp']):+.1f} pp "
                f"(Q1={100*safe_float(r['membership_rate_book_q1']):.1f}%, "
                f"Q4={100*safe_float(r['membership_rate_book_q4']):.1f}%)."
            )
    lines += [
        "",
        "## Miss audit",
        "",
        f"- High-priority fast/path-efficient alternate events not covered by the current 7A 75-within-30m definition: **{len(missed):,}** rows in the review table (capped at 250).",
        "",
        "## Interpretation discipline",
        "",
        "- This is an ontology robustness audit, not an alpha or profitability study.",
        "- Alternate definitions are hindsight labels built independently of Phase-7A / Phase-7I.",
        "- A high overlap means the current ontology is robust to reasonable alternate definitions.",
        "- A low overlap or economically obvious miss is a reason to inspect examples before freezing the event ontology further.",
        "- The volatility-normalized definition intentionally can select moves below 75 bps in quiet regimes because its target is max(50 bps, 3 x trailing 15m RV).",
        "- The 10-second anchor grid is an audit compromise between near-continuous coverage and tractable independent-window scanning; clustered event counts, not raw detection counts, should be used for event comparisons.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    _, pd, _, _ = require_stack()

    def rec(definition: str, anchor: int, hit: int, direction: int = +1):
        return {
            "detection_id": stable_id("SELF", definition, anchor, hit, direction),
            "definition": definition,
            "symbol": DEFAULT_SYMBOL,
            "direction": direction,
            "direction_label": "UP" if direction > 0 else "DOWN",
            "anchor_second": anchor,
            "anchor_time": iso_time(anchor),
            "anchor_date": datetime.fromtimestamp(anchor, tz=timezone.utc).date().isoformat(),
            "anchor_price": 100.0,
            "horizon_sec": 900 if definition == "FIRST_PASSAGE_75_15M" else 600,
            "target_bps": 100.0 if definition == "FIRST_PASSAGE_100_10M" else 75.0,
            "first_hit_second": hit,
            "first_hit_time": iso_time(hit),
            "seconds_to_hit": hit - anchor,
            "first_hit_price": 100.75,
            "terminal_return_bps": 80.0,
            "max_favorable_bps": 90.0,
            "prehit_mae_bps": 5.0,
            "trailing_rv15_bps": 20.0,
            "normalized_target_over_rv": 3.75,
            "anchor_contract": "SELF_TEST",
        }

    rows = []
    # Parent detections form one first impulse around hit ~= 1000.  The stricter
    # path-efficient subset deliberately skips middle anchors; independent
    # clustering could split it, but common-parent inheritance must not.
    for anchor, hit in ((500, 1000), (510, 1010), (520, 1020), (530, 1030), (540, 1040), (550, 1050)):
        rows.append(rec("FIRST_PASSAGE_75_15M", anchor, hit))
        rows.append(rec("FIRST_PASSAGE_75_10M", anchor, hit))
        if anchor in (500, 510, 540, 550):
            rows.append(rec("PATH_EFFICIENT_75_10M", anchor, hit))

    # A second distinct same-direction acceleration >180 seconds later.
    for anchor, hit in ((900, 1400), (910, 1410), (920, 1420)):
        rows.append(rec("FIRST_PASSAGE_75_15M", anchor, hit))
        rows.append(rec("FIRST_PASSAGE_75_10M", anchor, hit))
        rows.append(rec("FIRST_PASSAGE_100_10M", anchor, hit + 30))

    sample = pd.DataFrame(rows)
    impulses, events, nested = build_common_impulses_and_events(sample)
    fixed = impulses[impulses["impulse_family"] == "FIXED_75_PARENT"]
    assert len(fixed) == 2, fixed

    counts = events.groupby("definition")["impulse_id"].nunique().to_dict()
    assert counts["PATH_EFFICIENT_75_10M"] <= counts["FIRST_PASSAGE_75_10M"] <= counts["FIRST_PASSAGE_75_15M"]
    assert counts["FIRST_PASSAGE_100_10M"] <= counts["FIRST_PASSAGE_75_10M"]
    assert int(nested["subset_violation_count"].sum()) == 0, nested

    # Coverage-boundary regression: a valid 5-minute child may exist without a
    # 15-minute parent row.  It must still inherit a common impulse ID.
    edge = pd.DataFrame([
        rec("FIRST_PASSAGE_75_5M", 1600, 1700),
        rec("TERMINAL_75_5M", 1600, 1700),
    ])
    edge_impulses, edge_events, edge_nested = build_common_impulses_and_events(edge)
    assert len(edge_impulses) == 1, edge_impulses
    assert edge_events["impulse_id"].notna().all(), edge_events
    assert int(edge_nested["subset_violation_count"].sum()) == 0, edge_nested

    # Path-efficient detections from the first impulse must share ONE impulse
    # even though their raw anchors contain a gap.
    pe = events[events["definition"] == "PATH_EFFICIENT_75_10M"]
    assert pe["impulse_id"].nunique() == 1, pe

    print("SELF-TEST PASS")
    print("Validated union-parent boundary handling, common impulse clustering, non-transitive hit-span splitting, nested-definition inheritance, and subset-consistent event counts.")

def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7I.9 large-move ontology robustness audit")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--phase7i1-script", type=Path, default=DEFAULT_PHASE7I1_SCRIPT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--phase7a-root", type=Path, default=DEFAULT_PHASE7A_ROOT)
    p.add_argument("--phase7i1-root", type=Path, default=DEFAULT_PHASE7I1_ROOT)
    p.add_argument("--phase7i6-root", type=Path, default=DEFAULT_PHASE7I6_ROOT)
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
        raise SystemExit(f"Phase 7I.9 is frozen to symbol={DEFAULT_SYMBOL}; received {args.symbol}")
    if args.max_trade_stale_sec != DEFAULT_MAX_TRADE_STALE_SEC:
        raise SystemExit(
            f"Phase 7I.9 is frozen to max_trade_stale_sec={DEFAULT_MAX_TRADE_STALE_SEC}; "
            f"received {args.max_trade_stale_sec}"
        )

    phase7a = load_module(args.phase7a_script, "phase7a_frozen_for_7i9")
    phase7i1 = load_module(args.phase7i1_script, "phase7i1_frozen_for_7i9")
    if getattr(phase7a, "BUILDER_VERSION", None) != FROZEN_PHASE7A_VERSION:
        raise RuntimeError(
            f"Phase 7I.9 requires Phase-7A {FROZEN_PHASE7A_VERSION}; found "
            f"{getattr(phase7a, 'BUILDER_VERSION', None)!r}"
        )
    if getattr(phase7i1, "BUILDER_VERSION", None) != FROZEN_PHASE7I1_VERSION:
        raise RuntimeError(
            f"Phase 7I.9 requires Phase-7I.1 {FROZEN_PHASE7I1_VERSION}; found "
            f"{getattr(phase7i1, 'BUILDER_VERSION', None)!r}"
        )

    print("Building Phase 7I.9 large-move ontology robustness audit...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Frozen Phase-7A       : {phase7a.BUILDER_VERSION}")
    print(f"Frozen Phase-7I.1     : {phase7i1.BUILDER_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Dates                 : {start} to {end}")
    print(f"Independent anchors   : every {ANCHOR_CADENCE_SEC}s")
    print("Alternate definitions : fixed first-passage, fixed terminal, path-efficient, vol-normalized")
    print("Raw depth replay      : NONE")
    print("Research role         : event-ontology robustness audit; not alpha")

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
    sec, price, price_source = phase7a.build_catalog_price(frame, args.max_trade_stale_sec, "last")
    print(f"      rows={len(sec):,} price_source={price_source}")

    print("[2/8] Scanning alternate large-move definitions independently...")
    detections, nesting_checks, anchor_count = build_alternate_detections(sec, price, start, end, phase7a)
    print(f"      anchors={anchor_count:,} raw alternate detections={len(detections):,}")

    print("[3/8] Building common parent impulses and nested event labels...")
    impulses, events, nested_consistency = build_common_impulses_and_events(detections)
    print(f"      common impulses={len(impulses):,} labeled alternate events={len(events):,}")

    print("[4/8] Loading frozen Phase-7A / 7I.1 / 7I.6 tables...")
    frozen = load_frozen_tables(args.phase7a_root, args.phase7i1_root, args.phase7i6_root)
    physical = frozen["physical"]
    episodes = frozen["episodes"]
    candidates = frozen["candidates"]
    geometry = frozen["geometry"]
    analysis = frozen["analysis"]
    print(f"      physical legs={len(physical):,} causal candidates={len(candidates):,}")

    print("[5/8] Mapping common impulses to current physical legs and causal candidates...")
    mapped_impulses = map_alternate_events(impulses, physical, episodes, candidates)
    mapping_cols = [
        "impulse_id","mapped_phase7a_event","phase7a_event_id","phase7a_time_to_75_sec",
        "phase7a_time_to_100_sec","covered_by_current_7a_75_30m","covered_by_current_7a_100_30m",
        "mapped_causal_episode","causal_episode_id","covered_by_causal_35_candidate","candidate_id",
        "candidate_second","candidate_lead_to_alt_hit_sec","alternate_anchor_minus_candidate_sec",
    ]
    mapped_events = events.merge(mapped_impulses[mapping_cols], on="impulse_id", how="left")
    membership = build_candidate_membership(candidates, geometry, physical, mapped_events)

    print("[6/8] Auditing overlap, misses, and frozen book-failure robustness...")
    def_summary = definition_summary(mapped_events, physical, detections)
    coverage = coverage_report(mapped_events)
    mem_rates = candidate_membership_rates(membership)
    overlap = overlap_matrix(membership)
    book = load_book_state(analysis)
    book_robust, book_direction = book_failure_robustness(membership, book)
    missed = corrected_missed_fast_impulses(mapped_impulses, mapped_events)
    multiplicity = physical_leg_impulse_multiplicity(mapped_impulses, mapped_events)
    reaccels = reacceleration_impulses(mapped_impulses, mapped_events, multiplicity)

    print("[7/8] Building invariants and findings...")
    invariants = build_invariants(
        detections, mapped_events, membership, nesting_checks, anchor_count, book,
        impulses=impulses, nested_consistency=nested_consistency,
    )
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    candidate_hash = hashlib.sha256(
        "\n".join(sorted(candidates["candidate_id"].astype(str))).encode("utf-8")
    ).hexdigest()
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
        "anchor_cadence_sec": ANCHOR_CADENCE_SEC,
        "anchor_count": int(anchor_count),
        "alternate_detection_count": int(len(detections)),
        "common_impulse_count": int(len(mapped_impulses)),
        "fixed75_parent_impulse_count": int((mapped_impulses["impulse_family"] == "FIXED_75_PARENT").sum()),
        "alternate_event_label_count": int(len(mapped_events)),
        "physical_legs_with_multiple_fixed75_impulses": int((multiplicity["distinct_fixed75_impulses"] > 1).sum()) if len(multiplicity) else 0,
        "reacceleration_impulse_rows": int(len(reaccels)),
        "candidate_count": int(len(candidates)),
        "candidate_id_set_sha256": candidate_hash,
        "failed_invariants": int(len(failed)),
        "interpretation": (
            "Retrospective robustness audit of alternate large-move definitions. "
            "Alternate labels are independent of Phase-7A/7I construction; no trading or profitability claim."
        ),
    }
    run_config = {
        **summary,
        "fixed_horizons_sec": list(FIXED_HORIZONS_SEC),
        "first_passage_target_bps": FIRST_PASSAGE_TARGET_BPS,
        "path_efficient_horizon_sec": PATH_EFFICIENT_HORIZON_SEC,
        "path_efficient_max_prehit_mae_bps": PATH_EFFICIENT_MAX_PREHIT_MAE_BPS,
        "vol_lookback_sec": VOL_LOOKBACK_SEC,
        "vol_multiplier": VOL_MULTIPLIER,
        "vol_target_floor_bps": VOL_TARGET_FLOOR_BPS,
        "vol_horizon_sec": VOL_HORIZON_SEC,
        "cluster_max_anchor_gap_sec": CLUSTER_MAX_ANCHOR_GAP_SEC,
        "cluster_max_hit_gap_sec": CLUSTER_MAX_HIT_GAP_SEC,
        "common_parent_rule": "UNION_FIRST_PASSAGE_75_5M_10M_15M",
        "common_parent_source_definitions": list(COMMON_PARENT_SOURCE_DEFINITIONS),
        "common_parent_definitions": list(COMMON_PARENT_DEFINITIONS),
        "book_decision_horizon_sec": BOOK_DECISION_HORIZON_SEC,
        "definitions": list(DEFINITION_ORDER),
        "execution_root": str(args.execution_root),
        "phase7a_root": str(args.phase7a_root),
        "phase7i1_root": str(args.phase7i1_root),
        "phase7i6_root": str(args.phase7i6_root),
    }
    findings = findings_text(summary, def_summary, coverage, book_robust, missed)

    print("[8/8] Writing atomic datasets and reports...")
    atomic_write_parquet(detections, args.output_root / "alternate_detections.parquet")
    atomic_write_parquet(mapped_impulses, args.output_root / "alternate_impulses.parquet")
    atomic_write_parquet(mapped_events, args.output_root / "alternate_events.parquet")
    atomic_write_parquet(membership, args.output_root / "candidate_ontology_membership.parquet")
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", run_config)

    atomic_write_csv(def_summary, args.report_root / "ontology_definition_summary.csv")
    atomic_write_csv(coverage, args.report_root / "ontology_event_coverage.csv")
    atomic_write_csv(mem_rates, args.report_root / "ontology_candidate_membership_rates.csv")
    atomic_write_csv(overlap, args.report_root / "ontology_overlap_matrix.csv")
    atomic_write_csv(nested_consistency, args.report_root / "ontology_nested_consistency.csv")
    atomic_write_csv(multiplicity, args.report_root / "ontology_physical_leg_impulse_multiplicity.csv")
    atomic_write_csv(reaccels, args.report_root / "ontology_reacceleration_impulses.csv")
    atomic_write_csv(missed, args.report_root / "ontology_missed_fast_events.csv")
    atomic_write_csv(book_robust, args.report_root / "ontology_book_failure_robustness.csv")
    atomic_write_csv(book_direction, args.report_root / "ontology_book_failure_direction_robustness.csv")
    atomic_write_csv(invariants, args.report_root / "ontology_invariants.csv")
    atomic_write_text(args.report_root / "ontology_findings.md", findings)

    print("")
    print(f"Phase 7I.9 status: {status}")
    print(f"Alternate detections: {len(detections):,}")
    print(f"Common impulses     : {len(mapped_impulses):,}")
    print(f"Event labels        : {len(mapped_events):,}")
    print(f"Multi-impulse legs  : {int((multiplicity['distinct_fixed75_impulses'] > 1).sum()) if len(multiplicity) else 0:,}")
    print(f"Failed invariants   : {len(failed)}")
    print(f"Reports             : {args.report_root}")
    if not failed.empty:
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
