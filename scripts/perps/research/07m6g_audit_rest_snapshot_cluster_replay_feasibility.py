#!/usr/bin/env python3
"""
Phase 7M.6G — REST-snapshot clustered-replay feasibility audit
===============================================================

PLAIN-ENGLISH QUESTION
----------------------
Can the remaining Phase 7M.6F work be made materially faster by avoiding
full-day depth replay and instead bootstrapping short candidate clusters from
the latest stored Binance REST order-book snapshot before each cluster?

WHY THIS EXISTS
---------------
Phase 7M.6F is scientifically correct but operationally expensive because it
replays most of every candidate date even though each candidate only needs an
exact anchor plus at most 10 seconds of causal mechanics. The canonical store
already contains full Binance REST order-book snapshots used by the Phase-6A
LocalBook bootstrap contract. If those snapshots occur often enough, a future
fast-path can start near candidate clusters and skip long irrelevant stretches.

THIS PHASE IS ONLY A CHEAP FEASIBILITY AUDIT
--------------------------------------------
* No order-book levels are decoded.
* No LocalBook is constructed.
* No trades are loaded.
* No future/P75 outcomes are loaded.
* No candidate or scientific threshold is changed.
* Existing Phase 7M.6F v2 checkpoints are read only to identify completed days.

The script reads only scalar timestamp columns from canonical snapshot/depth
Parquet plus the frozen prospective candidate clock. It reports:
  - REST snapshot cadence by date;
  - nearest prior REST snapshot for every candidate;
  - candidate replay clusters obtained by merging overlapping intervals
      [latest prior REST snapshot, candidate_second + 11s];
  - the EXACT number of depth rows lying inside those merged intervals;
  - the fraction of full-day depth rows that clustered replay would touch;
  - remaining-work estimates after excluding already signed v2 checkpoints and
    Phase-7M.6B pilot dates that are tape-reused.

IMPORTANT
---------
This audit does NOT prove the clustered replay is scientifically equivalent.
If cadence/work reduction is favorable, the next phase must replay 2-3 already
completed dates with both methods and require candidate-level parity before the
fast path is allowed to build unfinished dates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pc = None
    pq = None


BUILDER_VERSION = "1.0.0"
SCHEMA_NAME = "prospective_25bp_rest_snapshot_cluster_replay_feasibility_v1"
FROZEN_PHASE7M3_SCHEMA = "prospective_25bp_exact_participant_interaction_v1"
FROZEN_POLICY = "EARLY_25BP_RESET5_RELEASE10_REARM60"
EXPECTED_CANDIDATES = 7067
EXPECTED_DAYS = 62
MAX_CAPTURE_AFTER_CANDIDATE_SEC = 11
EXPECTED_PILOT_DATES = {
    "2026-05-21", "2026-06-09", "2026-06-30", "2026-07-16"
}

# Operational recommendation bands only. They do not alter scientific data.
PROMISING_MAX_REPLAY_FRACTION = 0.50  # >=2x estimated depth-row reduction
MARGINAL_MAX_REPLAY_FRACTION = 0.80
MIN_PRIOR_SNAPSHOT_COVERAGE = 0.95

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_M3_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / FROZEN_PHASE7M3_SCHEMA
DEFAULT_V2_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "prospective_25bp_refill_redepletion_sequence_v2"
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_7M6B_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07m6b_build_refill_redepletion_sequence_pilot.py"
DEFAULT_PHASE6A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "06a_build_execution_foundation.py"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_prospective_25bp_rest_snapshot_cluster_replay_feasibility"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_module(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"Missing required repository script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6g-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7m6g-", suffix=path.suffix or ".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_scalar_ns(paths: Iterable[Path], *, symbol: str | None = None) -> np.ndarray:
    """Read only ts_recv_ns (+symbol when present) and return sorted int64 ns."""
    parts: list[np.ndarray] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        if "ts_recv_ns" not in names:
            raise RuntimeError(f"{path} lacks ts_recv_ns")
        cols = ["ts_recv_ns"] + (["symbol"] if symbol and "symbol" in names else [])
        for batch in pf.iter_batches(columns=cols, batch_size=131072):
            tbl = pa.Table.from_batches([batch])
            if symbol and "symbol" in cols:
                mask = pc.equal(tbl["symbol"], pa.scalar(symbol))
                tbl = tbl.filter(mask)
            if len(tbl) == 0:
                continue
            arr = pc.cast(tbl["ts_recv_ns"], pa.int64(), safe=False).to_numpy(zero_copy_only=False)
            parts.append(np.asarray(arr, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(parts)
    if len(out) > 1 and np.any(out[1:] < out[:-1]):
        out.sort()
    return out


def nearest_prior(snapshot_ns: np.ndarray, target_ns: int) -> int | None:
    i = int(np.searchsorted(snapshot_ns, int(target_ns), side="right")) - 1
    return int(snapshot_ns[i]) if i >= 0 else None


def merge_intervals(intervals: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    """Merge overlapping replay intervals; retain candidate IDs."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1], x[2]))
    out: list[dict[str, Any]] = []
    s, e, cid = intervals[0]
    ids = [cid]
    for ns, ne, ncid in intervals[1:]:
        if ns <= e:  # computational overlap; no scientific threshold
            e = max(e, ne)
            ids.append(ncid)
        else:
            out.append({"start_ns": int(s), "end_ns": int(e), "candidate_ids": ids})
            s, e, ids = ns, ne, [ncid]
    out.append({"start_ns": int(s), "end_ns": int(e), "candidate_ids": ids})
    return out


def rows_in_intervals(sorted_ts: np.ndarray, intervals: list[dict[str, Any]]) -> int:
    total = 0
    for x in intervals:
        lo = int(np.searchsorted(sorted_ts, int(x["start_ns"]), side="left"))
        hi = int(np.searchsorted(sorted_ts, int(x["end_ns"]), side="right"))
        total += max(0, hi - lo)
    return int(total)


def q(values: pd.Series, p: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.quantile(p)) if len(x) else math.nan


def signed_v2_checkpoint_exists(v2_root: Path, day: str) -> bool:
    p = v2_root / "checkpoints" / f"candidate_date={day}" / "manifest.json"
    if not p.exists():
        return False
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Feasibility audit only needs to know that the production builder committed
    # a PASS checkpoint. The production script itself performs signature reuse.
    return str(m.get("status", "")).upper() == "PASS" and str(m.get("schema_name", "")) == "prospective_25bp_refill_redepletion_sequence_v2"


def findings_text(summary: dict[str, Any]) -> str:
    return f"""# Phase 7M.6G — REST-snapshot clustered-replay feasibility audit

- Builder: **{BUILDER_VERSION}**
- Status: **{summary['status']}**
- Future outcomes loaded: **NO**
- LocalBook replay performed: **NO**
- Nested bid/ask/depth arrays decoded: **NO**

## Snapshot cadence

- Candidate dates: **{summary['candidate_dates']}**
- Total stored REST snapshots on candidate dates: **{summary['snapshot_count']}**
- Median snapshot gap: **{summary['median_snapshot_gap_sec']:.1f}s**
- P90 snapshot gap: **{summary['p90_snapshot_gap_sec']:.1f}s**
- Maximum snapshot gap: **{summary['max_snapshot_gap_sec']:.1f}s**
- Candidates with a prior REST snapshot: **{summary['prior_snapshot_coverage']:.2%}**
- Median candidate→prior-snapshot lag: **{summary['median_candidate_snapshot_lag_sec']:.1f}s**
- P90 candidate→prior-snapshot lag: **{summary['p90_candidate_snapshot_lag_sec']:.1f}s**

## Estimated replay work

Across all raw-eligible candidate dates:

- Full-day depth rows: **{summary['all_raw_full_depth_rows']:,}**
- Clustered depth rows: **{summary['all_raw_cluster_depth_rows']:,}**
- Clustered/full fraction: **{summary['all_raw_replay_fraction']:.2%}**
- Estimated depth-row reduction: **{summary['all_raw_reduction_pct']:.2%}**

For work still unfinished when this audit ran:

- Unfinished raw dates: **{summary['remaining_raw_dates']}**
- Full-day depth rows: **{summary['remaining_full_depth_rows']:,}**
- Clustered depth rows: **{summary['remaining_cluster_depth_rows']:,}**
- Clustered/full fraction: **{summary['remaining_replay_fraction']:.2%}**
- Estimated depth-row reduction: **{summary['remaining_reduction_pct']:.2%}**

## Interpretation

`CLUSTER_FAST_PATH_PROMISING` means the stored REST snapshot cadence appears capable of cutting scalar depth-row replay by at least half while covering at least 95% of candidates with an earlier snapshot. It does **not** prove scientific parity.

`CLUSTER_FAST_PATH_MARGINAL` means there is some operational saving, but not enough to assume the engineering complexity is worthwhile.

`CLUSTER_FAST_PATH_NOT_USEFUL` means cadence/coverage is unlikely to provide a worthwhile replay reduction.

## Mandatory next step if promising

Do **not** immediately use the clustered method on unfinished dates. First build a parity prototype on 2–3 already completed v2 dates and require candidate-level agreement for exact anchor recovery, refill amount/breadth, conservative replacement execution, and `replacement_penetrated` before promoting the fast path.
"""


def self_test() -> int:
    snaps = np.array([100, 200, 400], dtype=np.int64)
    assert nearest_prior(snaps, 199) == 100
    assert nearest_prior(snaps, 200) == 200
    assert nearest_prior(snaps, 99) is None
    xs = [(100, 150, "A"), (140, 180, "B"), (300, 320, "C")]
    m = merge_intervals(xs)
    assert len(m) == 2 and m[0]["candidate_ids"] == ["A", "B"]
    ts = np.arange(0, 500, 10, dtype=np.int64)
    # [100,180] gives 9 points inclusive; [300,320] gives 3.
    assert rows_in_intervals(ts, m) == 12
    print("PASS: Phase 7M.6G self-test (prior snapshot, overlap clustering, exact scalar depth-row accounting).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 7M.6G REST-snapshot clustered-replay feasibility audit")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--m3-root", type=Path, default=DEFAULT_M3_ROOT)
    p.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--phase7m6b-script", type=Path, default=DEFAULT_7M6B_SCRIPT)
    p.add_argument("--phase6a-script", type=Path, default=DEFAULT_PHASE6A_SCRIPT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()
    if pq is None or pa is None or pc is None:
        raise SystemExit("Phase 7M.6G requires pyarrow in the project .venv.")

    print("Phase 7M.6G — REST-snapshot clustered-replay feasibility audit")
    print("[1/5] Loading frozen 7M.3 candidate clock (no outcomes)...")
    phase7m6b = load_module(args.phase7m6b_script, "phase7m6b_for_7m6g")
    phase6a = load_module(args.phase6a_script, "phase6a_for_7m6g")
    _, candidates, mechanics = phase7m6b.load_m3(args.m3_root)
    clock = phase7m6b.derive_candidate_clock(candidates, mechanics)
    clock = clock[["candidate_id", "candidate_date", "candidate_second", "direction"]].copy()
    clock["candidate_id"] = clock["candidate_id"].astype(str)
    clock["candidate_date"] = clock["candidate_date"].astype(str)
    if len(clock) != EXPECTED_CANDIDATES or clock["candidate_date"].nunique() != EXPECTED_DAYS:
        raise RuntimeError(f"Frozen universe mismatch candidates={len(clock)} days={clock['candidate_date'].nunique()}")

    print("[2/5] Discovering canonical REST snapshot/depth files...")
    snapshot_days = phase6a.discover_family_days(args.canonical_root, "book_snapshots")
    depth_days = phase6a.discover_family_days(args.canonical_root, "depth_updates")

    candidate_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    gap_values: list[float] = []

    dates = sorted(clock["candidate_date"].unique().tolist())
    print("[3/5] Reading scalar timestamps and calculating exact clustered depth-row work...")
    for i, day in enumerate(dates, 1):
        ddate = pd.Timestamp(day).date()
        spaths = tuple(snapshot_days.get(ddate, ()))
        dpaths = tuple(depth_days.get(ddate, ()))
        day_clock = clock[clock["candidate_date"] == day].sort_values(["candidate_second", "candidate_id"]).copy()
        if not spaths:
            raise RuntimeError(f"No canonical REST snapshot files for candidate date {day}")
        if not dpaths:
            raise RuntimeError(f"No canonical depth files for candidate date {day}")

        snapshot_ns = read_scalar_ns(spaths, symbol=args.symbol)
        depth_ns = read_scalar_ns(dpaths, symbol=args.symbol)
        if len(snapshot_ns) == 0:
            raise RuntimeError(f"No {args.symbol} REST snapshot rows for {day}")
        if len(depth_ns) == 0:
            raise RuntimeError(f"No {args.symbol} depth rows for {day}")

        if len(snapshot_ns) > 1:
            gaps = np.diff(snapshot_ns).astype(np.float64) / 1e9
            gap_values.extend(gaps.tolist())
            med_gap = float(np.median(gaps)); p90_gap = float(np.quantile(gaps, .90)); max_gap = float(np.max(gaps))
        else:
            med_gap = p90_gap = max_gap = math.nan

        intervals: list[tuple[int, int, str]] = []
        prior_count = 0
        lags: list[float] = []
        for r in day_clock.itertuples(index=False):
            target_ns = int(r.candidate_second) * 1_000_000_000
            prior = nearest_prior(snapshot_ns, target_ns)
            if prior is None:
                lag = math.nan
            else:
                prior_count += 1
                lag = (target_ns - prior) / 1e9
                lags.append(lag)
                end_ns = (int(r.candidate_second) + MAX_CAPTURE_AFTER_CANDIDATE_SEC) * 1_000_000_000
                intervals.append((prior, end_ns, str(r.candidate_id)))
            candidate_rows.append({
                "candidate_id": str(r.candidate_id), "candidate_date": day,
                "candidate_second": int(r.candidate_second), "direction": int(r.direction),
                "prior_snapshot_found": prior is not None,
                "prior_snapshot_ts_recv_ns": prior,
                "prior_snapshot_lag_sec": lag,
            })

        merged = merge_intervals(intervals)
        cluster_depth_rows = rows_in_intervals(depth_ns, merged)
        full_depth_rows = int(len(depth_ns))
        completed = signed_v2_checkpoint_exists(args.v2_root, day)
        pilot_tape_reuse = day in EXPECTED_PILOT_DATES
        for ci, c in enumerate(merged, 1):
            lo = int(np.searchsorted(depth_ns, int(c["start_ns"]), side="left"))
            hi = int(np.searchsorted(depth_ns, int(c["end_ns"]), side="right"))
            cluster_rows.append({
                "candidate_date": day, "cluster_index": ci,
                "cluster_start_snapshot_ts_recv_ns": int(c["start_ns"]),
                "cluster_end_ns": int(c["end_ns"]),
                "cluster_span_sec": (int(c["end_ns"]) - int(c["start_ns"])) / 1e9,
                "candidate_count": len(c["candidate_ids"]),
                "candidate_ids": "|".join(c["candidate_ids"]),
                "depth_rows": max(0, hi - lo),
            })
        replay_fraction = cluster_depth_rows / full_depth_rows if full_depth_rows else math.nan
        day_rows.append({
            "candidate_date": day,
            "candidate_count": int(len(day_clock)),
            "snapshot_count": int(len(snapshot_ns)),
            "median_snapshot_gap_sec": med_gap,
            "p90_snapshot_gap_sec": p90_gap,
            "max_snapshot_gap_sec": max_gap,
            "prior_snapshot_candidate_count": int(prior_count),
            "prior_snapshot_coverage": prior_count / len(day_clock) if len(day_clock) else math.nan,
            "median_candidate_snapshot_lag_sec": float(np.median(lags)) if lags else math.nan,
            "p90_candidate_snapshot_lag_sec": float(np.quantile(lags, .90)) if lags else math.nan,
            "cluster_count": int(len(merged)),
            "full_day_depth_rows": full_depth_rows,
            "clustered_depth_rows": int(cluster_depth_rows),
            "clustered_replay_fraction": replay_fraction,
            "estimated_depth_row_reduction_pct": 1.0 - replay_fraction if math.isfinite(replay_fraction) else math.nan,
            "v2_checkpoint_complete": bool(completed),
            "pilot_tape_reuse_date": bool(pilot_tape_reuse),
            "remaining_raw_work": bool((not completed) and (not pilot_tape_reuse)),
        })
        print(f"  [{i:02d}/{len(dates)}] {day}: snapshots={len(snapshot_ns):4d} candidates={len(day_clock):3d} clusters={len(merged):3d} depth={cluster_depth_rows:,}/{full_depth_rows:,} ({replay_fraction:.1%}) complete={completed}")

        del snapshot_ns, depth_ns

    cand = pd.DataFrame(candidate_rows)
    days_df = pd.DataFrame(day_rows)
    clusters = pd.DataFrame(cluster_rows)

    print("[4/5] Summarizing all-date and remaining-work savings...")
    total_candidates = len(cand)
    prior_cov = float(cand["prior_snapshot_found"].mean()) if total_candidates else math.nan
    all_full = int(days_df[~days_df["pilot_tape_reuse_date"]]["full_day_depth_rows"].sum())
    all_cluster = int(days_df[~days_df["pilot_tape_reuse_date"]]["clustered_depth_rows"].sum())
    rem = days_df[days_df["remaining_raw_work"]].copy()
    rem_full = int(rem["full_day_depth_rows"].sum())
    rem_cluster = int(rem["clustered_depth_rows"].sum())
    all_frac = all_cluster / all_full if all_full else math.nan
    rem_frac = rem_cluster / rem_full if rem_full else 0.0

    if prior_cov >= MIN_PRIOR_SNAPSHOT_COVERAGE and (not rem_full or rem_frac <= PROMISING_MAX_REPLAY_FRACTION):
        status = "CLUSTER_FAST_PATH_PROMISING"
    elif prior_cov >= MIN_PRIOR_SNAPSHOT_COVERAGE and (not rem_full or rem_frac <= MARGINAL_MAX_REPLAY_FRACTION):
        status = "CLUSTER_FAST_PATH_MARGINAL"
    else:
        status = "CLUSTER_FAST_PATH_NOT_USEFUL"

    gaps_series = pd.Series(gap_values, dtype=float)
    lag_series = pd.to_numeric(cand["prior_snapshot_lag_sec"], errors="coerce")
    summary = {
        "builder_version": BUILDER_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "candidate_dates": len(days_df),
        "candidate_count": total_candidates,
        "snapshot_count": int(days_df["snapshot_count"].sum()),
        "median_snapshot_gap_sec": q(gaps_series, .50),
        "p90_snapshot_gap_sec": q(gaps_series, .90),
        "max_snapshot_gap_sec": float(gaps_series.max()) if len(gaps_series) else math.nan,
        "prior_snapshot_coverage": prior_cov,
        "median_candidate_snapshot_lag_sec": q(lag_series, .50),
        "p90_candidate_snapshot_lag_sec": q(lag_series, .90),
        "all_raw_full_depth_rows": all_full,
        "all_raw_cluster_depth_rows": all_cluster,
        "all_raw_replay_fraction": all_frac,
        "all_raw_reduction_pct": 1.0 - all_frac if math.isfinite(all_frac) else math.nan,
        "remaining_raw_dates": int(len(rem)),
        "remaining_full_depth_rows": rem_full,
        "remaining_cluster_depth_rows": rem_cluster,
        "remaining_replay_fraction": rem_frac,
        "remaining_reduction_pct": 1.0 - rem_frac if rem_full else 0.0,
        "future_outcomes_loaded": False,
        "localbook_replay_performed": False,
    }

    invariants = pd.DataFrame([
        {"check": "frozen_candidate_count", "passed": total_candidates == EXPECTED_CANDIDATES, "actual": total_candidates, "required": EXPECTED_CANDIDATES},
        {"check": "frozen_candidate_dates", "passed": len(days_df) == EXPECTED_DAYS, "actual": len(days_df), "required": EXPECTED_DAYS},
        {"check": "future_outcomes_loaded", "passed": True, "actual": False, "required": False},
        {"check": "localbook_replay_performed", "passed": True, "actual": False, "required": False},
        {"check": "prior_rest_snapshot_coverage", "passed": prior_cov >= MIN_PRIOR_SNAPSHOT_COVERAGE, "actual": prior_cov, "required": MIN_PRIOR_SNAPSHOT_COVERAGE},
    ])

    print("[5/5] Writing reports...")
    args.report_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.report_root / "snapshot_cadence_by_day.csv", days_df)
    atomic_csv(args.report_root / "candidate_prior_snapshot_audit.csv", cand)
    atomic_csv(args.report_root / "candidate_cluster_plan.csv", clusters)
    atomic_csv(args.report_root / "analysis_invariants.csv", invariants)
    atomic_text(args.report_root / "summary.json", json.dumps(summary, indent=2, sort_keys=True))
    atomic_text(args.report_root / "findings.md", findings_text(summary))

    print(f"Status: {status}")
    print(f"Prior REST snapshot coverage: {prior_cov:.2%}")
    print(f"Remaining full-day depth rows: {rem_full:,}")
    print(f"Remaining clustered depth rows: {rem_cluster:,}")
    print(f"Estimated remaining depth-row reduction: {(1.0-rem_frac):.2%}" if rem_full else "No unfinished raw dates detected.")
    print(f"Reports: {args.report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
