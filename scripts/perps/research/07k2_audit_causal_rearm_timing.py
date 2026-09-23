#!/usr/bin/env python3
"""
Phase 7K.2 — Causal re-arm timing audit
=======================================

Purpose
-------
Test one hypothesis only: Phase-7K.1's 180-second pre-reset cooldown may be
suppressing fast, genuinely distinct re-acceleration impulses.

Frozen mechanics
----------------
* reset threshold: 5 bps
* fresh release threshold: 10 bps
* candidate threshold: 35 bps
* candidate generation: frozen Phase-7K.1 v1.0.2 state machine
* hindsight association: corrected Phase-7K.1 v1.0.2 mapping
* hindsight impulse universe: frozen Phase-7I.9.2 fixed-75 common impulses
* no raw depth replay
* no raw trade replay
* no ML

Only experimental variable
--------------------------
TEST_COOLDOWN_0S
TEST_COOLDOWN_60S
TEST_COOLDOWN_120S

Benchmark
---------
BENCHMARK_COOLDOWN_180S
OLD_7I1_FROZEN

The script does NOT automatically choose a winning cooldown. A useful branch
must improve corrected re-acceleration coverage without disproportionate
candidate proliferation or obvious continuation-quality dilution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "causal_rearm_timing_audit_v1"
DEFAULT_SYMBOL = "BTCUSDT"

FROZEN_PHASE7A_VERSION = "1.0.0"
FROZEN_PHASE7K1_VERSION = "1.0.2"
FROZEN_PHASE7I9_VERSION = "1.0.2"

RESET_BPS = 5.0
RELEASE_BPS = 10.0
CANDIDATE_BPS = 35.0
EXPERIMENTAL_COOLDOWNS = (0, 60, 120)
BENCHMARK_COOLDOWN = 180
POLICY_COOLDOWNS = {
    "TEST_COOLDOWN_0S": 0,
    "TEST_COOLDOWN_60S": 60,
    "TEST_COOLDOWN_120S": 120,
    "BENCHMARK_COOLDOWN_180S": 180,
}
OLD_POLICY = "OLD_7I1_FROZEN"
DEFAULT_MAX_TRADE_STALE_SEC = 5

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts/perps/research/07a_build_large_move_event_catalog.py"
DEFAULT_PHASE7K1_SCRIPT = REPO_ROOT / "scripts/perps/research/07k1_build_causal_repeatable_expansion_impulse_candidate_catalog.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data/derived/perps/execution_foundation_v1"
DEFAULT_PHASE7I1_ROOT = REPO_ROOT / "data/derived/perps/research/causal_large_move_candidate_catalog_v1"
DEFAULT_PHASE7I9_ROOT = REPO_ROOT / "data/derived/perps/research/causal_large_move_ontology_robustness_audit_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/derived/perps/research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/perps_causal_rearm_timing_audit"


def require_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Phase 7K.2 requires numpy and pandas. Activate the project .venv first.") from exc
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
        raise SystemExit("Phase 7K.2 Parquet I/O requires pyarrow. Activate the project .venv first.")
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


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7k2-", suffix=path.suffix, dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7k2-", suffix=".csv", dir=str(path.parent))
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
    fd, tmp = tempfile.mkstemp(prefix="tmp7k2-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def generate_policy_candidates(phase7k1, sec, price, policy_id: str, cooldown_sec: int, start_epoch: int, end_epoch: int):
    """Reuse frozen 7K.1 while changing only MIN_REARM_SEPARATION_SEC."""
    original = int(phase7k1.MIN_REARM_SEPARATION_SEC)
    try:
        phase7k1.MIN_REARM_SEPARATION_SEC = int(cooldown_sec)
        out = phase7k1.build_candidates_for_policy(
            sec, price, policy_id, RESET_BPS, start_epoch, end_epoch
        )
    finally:
        phase7k1.MIN_REARM_SEPARATION_SEC = original
    if len(out):
        out["cooldown_sec"] = int(cooldown_sec)
        out["experimental_role"] = "TEST" if cooldown_sec in EXPERIMENTAL_COOLDOWNS else "BENCHMARK"
    return out


def policy_summary(candidates, phase7k1):
    _, pd, _, _ = require_stack()
    out = phase7k1.policy_summary(candidates).copy()
    out["cooldown_sec"] = out["policy_id"].map(POLICY_COOLDOWNS)
    n180 = int(out.loc[out["policy_id"] == "BENCHMARK_COOLDOWN_180S", "candidates"].iloc[0])
    out["candidate_change_vs_180"] = out["candidates"].astype(float) / float(n180) - 1.0
    return out.sort_values("cooldown_sec", kind="mergesort").reset_index(drop=True)


def incremental_coverage(impulses, mapping):
    _, pd, _, _ = require_stack()
    reacc_ids = set(
        impulses.loc[impulses["impulse_ordinal_in_physical_leg"] > 1, "impulse_id"].astype(str)
    )
    covered = {}
    for policy in [*POLICY_COOLDOWNS.keys(), OLD_POLICY]:
        g = mapping[(mapping["policy_id"] == policy) & mapping["mapped"].fillna(False).astype(bool)]
        covered[policy] = set(g["impulse_id"].astype(str)) & reacc_ids

    b180 = covered["BENCHMARK_COOLDOWN_180S"]
    old = covered.get(OLD_POLICY, set())
    rows = []
    for policy, cooldown in POLICY_COOLDOWNS.items():
        s = covered[policy]
        rows.append({
            "policy_id": policy,
            "cooldown_sec": cooldown,
            "reacceleration_impulses": len(reacc_ids),
            "covered_reaccelerations": len(s),
            "coverage_rate": len(s) / len(reacc_ids) if reacc_ids else math.nan,
            "incremental_vs_180": len(s - b180),
            "lost_vs_180": len(b180 - s),
            "net_vs_180": len(s) - len(b180),
            "incremental_vs_old_7i1": len(s - old),
            "lost_vs_old_7i1": len(old - s),
            "net_vs_old_7i1": len(s) - len(old),
            "union_with_old_7i1": len(s | old),
            "union_with_old_7i1_rate": len(s | old) / len(reacc_ids) if reacc_ids else math.nan,
        })
    return pd.DataFrame(rows).sort_values("cooldown_sec", kind="mergesort").reset_index(drop=True)


def cooldown_diagnostics(candidates):
    _, pd, _, _ = require_stack()
    rows = []
    for policy, g in candidates.groupby("policy_id", sort=False):
        re = g[g["origin_kind"] == "REARM_RESET"].copy()
        gap = pd.to_numeric(re["seconds_since_prior_same_direction_candidate"], errors="coerce")
        rows.append({
            "policy_id": policy,
            "cooldown_sec": int(POLICY_COOLDOWNS[policy]),
            "rearm_candidates": len(re),
            "min_rearm_gap_sec": float(gap.min()) if len(gap) else math.nan,
            "p10_rearm_gap_sec": float(gap.quantile(.10)) if len(gap) else math.nan,
            "median_rearm_gap_sec": float(gap.median()) if len(gap) else math.nan,
            "share_rearm_gap_lt_180s": float((gap < 180).mean()) if len(gap) else math.nan,
            "share_rearm_gap_lt_300s": float((gap < 300).mean()) if len(gap) else math.nan,
        })
    return pd.DataFrame(rows).sort_values("cooldown_sec", kind="mergesort").reset_index(drop=True)


def build_invariants(candidates, outcomes, mapping, impulses, phase7k1):
    _, pd, _, _ = require_stack()
    rows = []
    def add(name, ok, observed, expected):
        rows.append({"invariant": name, "status": "PASS" if ok else "FAIL", "observed": observed, "expected": expected})

    add("frozen_7k1_version", phase7k1.BUILDER_VERSION == FROZEN_PHASE7K1_VERSION, phase7k1.BUILDER_VERSION, FROZEN_PHASE7K1_VERSION)
    add("candidate_ids_unique", not candidates["candidate_id"].duplicated().any(), int(candidates["candidate_id"].duplicated().sum()), 0)
    add("outcomes_cover_candidates", set(candidates["candidate_id"].astype(str)) == set(outcomes["candidate_id"].astype(str)), len(outcomes), len(candidates))
    add("fixed_reset_5bp", bool((pd.to_numeric(candidates["reset_threshold_bps"], errors="coerce") == RESET_BPS).all()), sorted(set(pd.to_numeric(candidates["reset_threshold_bps"], errors="coerce").dropna())), [RESET_BPS])
    add("fixed_release_10bp", bool((pd.to_numeric(candidates["release_threshold_bps"], errors="coerce") == RELEASE_BPS).all()), sorted(set(pd.to_numeric(candidates["release_threshold_bps"], errors="coerce").dropna())), [RELEASE_BPS])
    add("fixed_candidate_35bp", bool((pd.to_numeric(candidates["candidate_threshold_bps"], errors="coerce") == CANDIDATE_BPS).all()), sorted(set(pd.to_numeric(candidates["candidate_threshold_bps"], errors="coerce").dropna())), [CANDIDATE_BPS])

    re = candidates[candidates["origin_kind"] == "REARM_RESET"].copy()
    gaps = pd.to_numeric(re["seconds_since_prior_same_direction_candidate"], errors="coerce")
    cds = pd.to_numeric(re["cooldown_sec"], errors="coerce")
    add("rearms_respect_branch_cooldown", bool((gaps >= cds).all()) if len(re) else True, int((gaps < cds).sum()) if len(re) else 0, 0)

    mr = mapping[
        mapping["mapped"].fillna(False).astype(bool)
        & (pd.to_numeric(mapping["impulse_ordinal_in_physical_leg"], errors="coerce") > 1)
    ].copy()
    if len(mr):
        prev = pd.to_numeric(mr["previous_impulse_hit_second"], errors="coerce")
        origin = pd.to_numeric(mr["candidate_origin_second"], errors="coerce")
        cand = pd.to_numeric(mr["candidate_second"], errors="coerce")
        hit = pd.to_numeric(mr["impulse_hit_second"], errors="coerce")
        release = pd.to_numeric(mr["candidate_release_second"], errors="coerce")
        have_release = mr["candidate_release_available"].fillna(False).astype(bool)
        ok = (prev < origin) & (origin <= cand) & (cand <= hit) & (~have_release | ((origin <= release) & (release <= cand)))
        add("mapped_reaccels_fresh_after_previous_impulse", bool(ok.all()), int((~ok).sum()), 0)

    add("expected_fixed75_impulses", len(impulses) == 654, len(impulses), 654)
    rn = int((impulses["impulse_ordinal_in_physical_leg"] > 1).sum())
    add("expected_reacceleration_impulses", rn == 223, rn, 223)
    return pd.DataFrame(rows)


def findings_text(status, summary_df, base_rates_df, coverage_df, incremental_df):
    lines = [
        "# Phase 7K.2 — Causal re-arm timing audit", "",
        f"- Builder: **{BUILDER_VERSION}**", f"- Status: **{status}**",
        "- Reset / release / candidate: **5 / 10 / 35 bps fixed**",
        "- Experimental cooldowns: **0 / 60 / 120 seconds**",
        "- Benchmark cooldown: **180 seconds**", "",
        "## Candidate proliferation", "",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"- {r['policy_id']}: {int(r['candidates']):,} candidates; "
            f"{100*safe_float(r['candidate_change_vs_180']):+.1f}% vs 180s; "
            f"median/day={safe_float(r['median_candidates_per_day']):.1f}."
        )
    lines += ["", "## Corrected re-acceleration coverage", ""]
    for _, r in incremental_df.iterrows():
        lines.append(
            f"- {r['policy_id']}: {int(r['covered_reaccelerations'])}/{int(r['reacceleration_impulses'])} "
            f"({100*safe_float(r['coverage_rate']):.1f}%); net vs 180s={int(r['net_vs_180']):+d}; "
            f"net vs old7I.1={int(r['net_vs_old_7i1']):+d}."
        )
    old = coverage_df[(coverage_df["policy_id"] == OLD_POLICY) & (coverage_df["segment"] == "REACCEL_IMPULSES")]
    if len(old):
        r = old.iloc[0]
        lines.append(f"- OLD_7I1_FROZEN: {int(r['covered_impulses'])}/{int(r['impulses'])} ({100*safe_float(r['coverage_rate']):.1f}%).")
    lines += ["", "## Outcome base rates", ""]
    for _, r in base_rates_df.iterrows():
        lines.append(
            f"- {r['policy_id']}: P75<=5m={100*safe_float(r['p75_5m']):.1f}%; "
            f"P75<=10m={100*safe_float(r['p75_10m']):.1f}%; P75<=30m={100*safe_float(r['p75_30m']):.1f}%; "
            f"path-efficient75/10={100*safe_float(r['path_efficient_75_10m_rate']):.1f}%."
        )
    lines += [
        "", "## Decision discipline", "",
        "- Do not select the shortest cooldown merely because it maximizes coverage.",
        "- Prefer material re-acceleration coverage improvement with controlled candidate growth.",
        "- Base-rate dilution is evidence that earlier re-arming mostly creates noisy/duplicate attempts.",
        "- No raw-depth replay until a causal impulse ontology is explicitly frozen.", "",
    ]
    return "\n".join(lines)


def self_test(phase7k1_script: Path) -> None:
    np, _, _, _ = require_stack()
    phase7k1 = load_module(phase7k1_script, "phase7k1_for_7k2_selftest")
    if phase7k1.BUILDER_VERSION != FROZEN_PHASE7K1_VERSION:
        raise RuntimeError(f"Self-test requires 7K.1 {FROZEN_PHASE7K1_VERSION}; found {phase7k1.BUILDER_VERSION}")

    n = 700
    sec = np.arange(1_700_100_000, 1_700_100_000 + n, dtype=np.int64)
    price = np.full(n, 100.0, dtype=float)
    price[:40] = 100.0
    price[40:101] = np.linspace(100.0, 100.40, 61)
    price[101:131] = np.linspace(100.40, 100.33, 30)  # early >5bp reset
    price[131:231] = np.linspace(100.33, 100.75, 100)
    price[231:] = 100.75
    start_epoch, end_epoch = int(sec[0]), int(sec[-1] + 1)

    built = {}
    for policy, cooldown in POLICY_COOLDOWNS.items():
        built[policy] = generate_policy_candidates(phase7k1, sec, price, policy, cooldown, start_epoch, end_epoch)
    up = {p: int((g["direction"] == 1).sum()) if len(g) else 0 for p, g in built.items()}
    assert up["TEST_COOLDOWN_0S"] >= up["BENCHMARK_COOLDOWN_180S"]
    assert up["TEST_COOLDOWN_60S"] >= up["BENCHMARK_COOLDOWN_180S"]
    assert int(phase7k1.MIN_REARM_SEPARATION_SEC) == 180
    print("SELF-TEST PASS")
    print("Validated isolated 0/60/120/180s cooldown sensitivity while keeping frozen 5bp reset, 10bp release and 35bp candidate mechanics unchanged.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7K.2 causal re-arm timing audit")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--phase7k1-script", type=Path, default=DEFAULT_PHASE7K1_SCRIPT)
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
        self_test(args.phase7k1_script)
        return 0

    np, pd, _, _ = require_stack()
    start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    if start > end:
        raise SystemExit("--start-date must be <= --end-date")
    if args.symbol != DEFAULT_SYMBOL:
        raise SystemExit(f"Phase 7K.2 is frozen to {DEFAULT_SYMBOL}; received {args.symbol}")
    if args.max_trade_stale_sec != DEFAULT_MAX_TRADE_STALE_SEC:
        raise SystemExit(f"Phase 7K.2 is frozen to max_trade_stale_sec={DEFAULT_MAX_TRADE_STALE_SEC}")

    phase7a = load_module(args.phase7a_script, "phase7a_frozen_for_7k2")
    phase7k1 = load_module(args.phase7k1_script, "phase7k1_frozen_for_7k2")
    if phase7a.BUILDER_VERSION != FROZEN_PHASE7A_VERSION:
        raise RuntimeError(f"Requires Phase-7A {FROZEN_PHASE7A_VERSION}; found {phase7a.BUILDER_VERSION}")
    if phase7k1.BUILDER_VERSION != FROZEN_PHASE7K1_VERSION:
        raise RuntimeError(f"Requires Phase-7K.1 {FROZEN_PHASE7K1_VERSION}; found {phase7k1.BUILDER_VERSION}")

    print("Building Phase 7K.2 causal re-arm timing audit...")
    print(f"Builder version       : {BUILDER_VERSION}")
    print(f"Frozen Phase-7K.1     : {phase7k1.BUILDER_VERSION}")
    print(f"Required Phase-7I.9   : {FROZEN_PHASE7I9_VERSION}")
    print(f"Symbol                : {args.symbol}")
    print(f"Dates                 : {start} to {end}")
    print("Reset / release / cand: 5 / 10 / 35 bp FIXED")
    print(f"Experimental cooldowns: {EXPERIMENTAL_COOLDOWNS}")
    print(f"Benchmark cooldown    : {BENCHMARK_COOLDOWN}s")
    print("Raw depth replay      : NONE")
    print("Raw trade replay      : NONE")

    print("[1/8] Loading Phase-6A one-second execution prices...")
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

    start_epoch = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_epoch = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())

    print("[2/8] Generating cooldown branches...")
    frames = []
    for policy, cooldown in POLICY_COOLDOWNS.items():
        c = generate_policy_candidates(phase7k1, sec, price, policy, cooldown, start_epoch, end_epoch)
        print(f"      {policy}: candidates={len(c):,} rearm={(c['origin_kind'] == 'REARM_RESET').sum() if len(c) else 0:,}")
        frames.append(c)
    runtime_candidates = pd.concat(frames, ignore_index=True).sort_values(["policy_id", "candidate_second", "direction"], kind="mergesort").reset_index(drop=True)

    print("[3/8] Building physically separate future outcome labels...")
    outcomes = phase7k1.build_candidate_outcomes(runtime_candidates, sec, price)
    print(f"      outcome rows={len(outcomes):,}")

    print("[4/8] Loading frozen Phase-7I.9.2 common impulses...")
    audit_meta, impulses_raw, events = phase7k1.load_7i9(args.phase7i9_root)
    if str(audit_meta.get("builder_version")) != FROZEN_PHASE7I9_VERSION:
        raise RuntimeError(f"Requires Phase-7I.9 {FROZEN_PHASE7I9_VERSION}; found {audit_meta.get('builder_version')}")
    impulses = phase7k1.impulse_definition_flags(phase7k1.add_impulse_ordinals(impulses_raw), events)
    reacc_n = int((impulses["impulse_ordinal_in_physical_leg"] > 1).sum())
    print(f"      fixed75 impulses={len(impulses):,} reaccelerations={reacc_n:,}")

    print("[5/8] Applying corrected fresh-origin impulse mapping...")
    mappings = []
    report_policies = list(POLICY_COOLDOWNS)
    for policy in POLICY_COOLDOWNS:
        m = phase7k1.map_candidates_to_impulses(runtime_candidates, impulses, policy)
        mappings.append(m)
        reacc_cov = int(m.loc[m["impulse_ordinal_in_physical_leg"] > 1, "mapped"].sum())
        print(f"      {policy}: all={int(m['mapped'].sum()):,}/{len(impulses):,} reaccel={reacc_cov:,}/{reacc_n:,}")

    old = phase7k1.load_old_7i1_candidates(args.phase7i1_root)
    if len(old):
        m_old = phase7k1.map_candidates_to_impulses(old, impulses, OLD_POLICY)
        mappings.append(m_old)
        report_policies.append(OLD_POLICY)
        old_reacc = int(m_old.loc[m_old["impulse_ordinal_in_physical_leg"] > 1, "mapped"].sum())
        print(f"      {OLD_POLICY}: all={int(m_old['mapped'].sum()):,}/{len(impulses):,} reaccel={old_reacc:,}/{reacc_n:,}")
    mapping = pd.concat(mappings, ignore_index=True)

    print("[6/8] Building reports...")
    public_candidates = runtime_candidates.drop(columns=["_price_index"]).copy()
    summary_df = policy_summary(public_candidates, phase7k1)
    base_rates_df = phase7k1.base_rates(public_candidates, outcomes)
    coverage_df = phase7k1.coverage_report(impulses, mapping, report_policies)
    ordinal_df = phase7k1.coverage_by_ordinal(impulses, mapping, report_policies)
    incremental_df = incremental_coverage(impulses, mapping)
    spacing_df = phase7k1.spacing_report(public_candidates)
    cooldown_df = cooldown_diagnostics(public_candidates)

    print("[7/8] Running invariants...")
    invariants = build_invariants(public_candidates, outcomes, mapping, impulses, phase7k1)
    failed = invariants[invariants["status"] == "FAIL"]
    status = "PASS" if failed.empty else "FAIL"

    hashes = {
        policy: hashlib.sha256("\n".join(sorted(public_candidates.loc[public_candidates["policy_id"] == policy, "candidate_id"].astype(str))).encode("utf-8")).hexdigest()
        for policy in POLICY_COOLDOWNS
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
        "reset_bps": RESET_BPS,
        "release_bps": RELEASE_BPS,
        "candidate_bps": CANDIDATE_BPS,
        "experimental_cooldowns_sec": list(EXPERIMENTAL_COOLDOWNS),
        "benchmark_cooldown_sec": BENCHMARK_COOLDOWN,
        "candidate_count_by_policy": {p: int((public_candidates["policy_id"] == p).sum()) for p in POLICY_COOLDOWNS},
        "candidate_id_set_sha256_by_policy": hashes,
        "fixed75_impulse_count": len(impulses),
        "reacceleration_impulse_count": reacc_n,
        "scientific_state": "TIMING_AUDIT_NOT_YET_FROZEN",
        "failed_invariants": len(failed),
    }
    run_config = {
        **summary,
        "frozen_phase7k1_builder_version": phase7k1.BUILDER_VERSION,
        "frozen_phase7i9_builder_version": audit_meta.get("builder_version"),
        "policy_cooldowns_sec": POLICY_COOLDOWNS,
        "mapping_rule": "7K.1.2 corrected fresh-origin mapping",
        "execution_root": str(args.execution_root),
        "phase7i1_root": str(args.phase7i1_root),
        "phase7i9_root": str(args.phase7i9_root),
    }
    findings = findings_text(status, summary_df, base_rates_df, coverage_df, incremental_df)

    print("[8/8] Writing atomic datasets and reports...")
    atomic_write_parquet(public_candidates, args.output_root / "candidates.parquet")
    atomic_write_parquet(outcomes, args.output_root / "candidate_outcomes.parquet")
    atomic_write_parquet(mapping, args.output_root / "candidate_to_impulse_mapping.parquet")
    atomic_write_json(args.output_root / "metadata/latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata/run_config.json", run_config)
    atomic_write_csv(summary_df, args.report_root / "rearm_timing_policy_summary.csv")
    atomic_write_csv(base_rates_df, args.report_root / "rearm_timing_base_rates.csv")
    atomic_write_csv(coverage_df, args.report_root / "rearm_timing_7i9_coverage.csv")
    atomic_write_csv(ordinal_df, args.report_root / "rearm_timing_coverage_by_ordinal.csv")
    atomic_write_csv(incremental_df, args.report_root / "rearm_timing_incremental_coverage.csv")
    atomic_write_csv(spacing_df, args.report_root / "rearm_timing_candidate_spacing.csv")
    atomic_write_csv(cooldown_df, args.report_root / "rearm_timing_cooldown_diagnostics.csv")
    atomic_write_csv(mapping, args.report_root / "rearm_timing_mapping_audit.csv")
    atomic_write_csv(invariants, args.report_root / "rearm_timing_invariants.csv")
    atomic_write_text(args.report_root / "rearm_timing_findings.md", findings)

    print("")
    print(f"Phase 7K.2 status: {status}")
    for _, r in incremental_df.iterrows():
        n = int((public_candidates["policy_id"] == r["policy_id"]).sum())
        print(f"{r['policy_id']:24s} candidates={n:5,d} reaccel={int(r['covered_reaccelerations']):3d}/{int(r['reacceleration_impulses'])}")
    print(f"Failed invariants: {len(failed)}")
    print("Scientific state : TIMING_AUDIT_NOT_YET_FROZEN")
    print(f"Reports          : {args.report_root}")
    if len(failed):
        print("")
        print(failed.to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
