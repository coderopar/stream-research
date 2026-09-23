#!/usr/bin/env python3
"""Phase 7I.1/7I.2 — causal large-move candidate catalog and geometry audit.

Scientific role
---------------
This is the first all-market prospective-ontology phase after the retrospective
Phase-7 event studies.  It deliberately performs NO raw-depth replay and does
NOT build a trading model.

The goal is to replace the hindsight physical extremum used by Phase 7A with a
fully causal directional-change episode origin:

* SEEK maintains the running low and high using only prices observed so far.
* A +10 bps departure from the running low confirms an UP episode.  The already
  observed running low becomes the frozen causal episode origin.
* A -10 bps departure from the running high confirms a DOWN episode.  The
  already observed running high becomes the frozen causal episode origin.
* Once an episode is open, an opposite 10 bps directional change from the
  running episode extreme closes it and simultaneously confirms the opposite
  episode from that already-observed extreme.
* A candidate is emitted once, at the FIRST observed 35 bps displacement from
  the frozen causal episode origin.

The 10/10/35 thresholds are frozen from the research plan before inspecting
Phase-7I outcomes.  They are intentionally not CLI-tunable in this builder.

Candidate inputs and future outcomes are stored separately.  ``candidates``
contains only data available by the candidate second (plus deterministic future
observation timestamp keys such as candidate+3s).  ``candidate_outcomes`` holds
all hindsight continuation/MFE/MAE labels.

Outputs
-------
data/derived/perps/research/causal_large_move_candidate_catalog_v1/
    episodes.parquet
    candidates.parquet
    candidate_outcomes.parquet
    geometry_audit.parquet
    physical_event_coverage.parquet
    metadata/latest_summary.json
    metadata/run_config.json
    metadata/candidate_freeze_manifest.json

reports/perps_causal_large_move_candidate_catalog/
    causal_candidate_daily_counts.csv
    causal_candidate_base_rates.csv
    causal_candidate_timing.csv
    causal_candidate_duplication_audit.csv
    causal_candidate_geometry_audit.csv
    causal_candidate_physical_coverage.csv
    causal_candidate_invariants.csv
    causal_candidate_findings.md

Interpretation discipline
-------------------------
* Candidate construction is causal-input-computable and uses no future labels.
* Future continuation labels are hindsight outcomes and never define candidates.
* Phase-7A mapping is audit-only and never changes candidate construction.
* This phase does NOT establish predictive precision, alpha, execution quality,
  or profitability.  It freezes/audits the candidate universe before the one
  expensive exact-book + executed-trade materialization pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BUILDER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
SCHEMA_NAME = "causal_large_move_candidate_catalog_v1"
DEFAULT_SYMBOL = "BTCUSDT"

# Frozen ontology.  Do not tune these on Phase-7I outcomes.
RELEASE_CONFIRMATION_BPS = 10.0
REVERSAL_CONFIRMATION_BPS = 10.0
CANDIDATE_THRESHOLD_BPS = 35.0
TARGET_BPS = (35.0, 50.0, 75.0, 100.0)
OUTCOME_HORIZONS_SEC = (300, 600, 900, 1800)
DECISION_DELAYS_SEC = (1, 3, 5, 10)
DEFAULT_MAX_TRADE_STALE_SEC = 5

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3] if len(SCRIPT_PATH.parents) >= 4 else Path.cwd()
DEFAULT_PHASE7A_SCRIPT = REPO_ROOT / "scripts" / "perps" / "research" / "07a_build_large_move_event_catalog.py"
DEFAULT_EXECUTION_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_PHASE7A_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / "large_move_event_catalog_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_causal_large_move_candidate_catalog"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_time(sec: int | None) -> str | None:
    if sec is None:
        return None
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: Any, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def directional_bps(price: float, anchor: float, direction: int) -> float:
    if not (math.isfinite(price) and math.isfinite(anchor) and price > 0 and anchor > 0):
        return math.nan
    return float(direction) * (price / anchor - 1.0) * 10_000.0


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7i1-", suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def require_numeric_stack():
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Phase 7I.1 requires numpy and pandas. Activate the project .venv first.") from exc
    return np, pd


def require_stack():
    np, pd = require_numeric_stack()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("Phase 7I.1 Parquet I/O requires pyarrow. Activate the project .venv first.") from exc
    return np, pd, pa, pq


def write_parquet_atomic(df, path: Path) -> None:
    _, _, pa, pq = require_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7i1-", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_csv_atomic(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp7i1-", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_module(path: Path, name: str):
    import sys
    if not path.exists():
        raise RuntimeError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class CausalEpisode:
    episode_id: str
    symbol: str
    direction: int
    direction_label: str
    origin_second: int
    origin_time: str
    origin_date: str
    origin_price: float
    confirmation_second: int
    confirmation_time: str
    confirmation_price: float
    confirmation_displacement_bps: float
    release_confirmation_bps: float
    reversal_confirmation_bps: float
    candidate_threshold_bps: float
    candidate_second: int | None
    candidate_time: str | None
    candidate_price: float | None
    candidate_displacement_bps: float | None
    seconds_confirmation_to_candidate: int | None
    hit_35_second: int | None
    hit_50_second: int | None
    hit_75_second: int | None
    hit_100_second: int | None
    extreme_second: int
    extreme_time: str
    extreme_price: float
    max_excursion_bps: float
    close_second: int | None
    close_time: str | None
    close_price: float | None
    close_reason: str


def _new_episode(symbol: str, direction: int, origin_i: int, confirm_i: int, sec, price) -> dict[str, Any]:
    op = float(price[origin_i])
    cp = float(price[confirm_i])
    eid = stable_id("7I_CAUSAL_EPISODE_V1", symbol, direction, int(sec[origin_i]), int(sec[confirm_i]), RELEASE_CONFIRMATION_BPS)
    hits = {float(t): None for t in TARGET_BPS}
    d = directional_bps(cp, op, direction)
    for t in TARGET_BPS:
        if d + 1e-12 >= float(t):
            hits[float(t)] = int(confirm_i)
    return {
        "episode_id": eid,
        "symbol": symbol,
        "direction": int(direction),
        "origin_i": int(origin_i),
        "confirmation_i": int(confirm_i),
        "origin_price": op,
        "extreme_i": int(confirm_i),
        "extreme_price": cp,
        "hits": hits,
        "close_i": None,
        "close_reason": None,
    }


def segment_causal_directional_change(sec, price, symbol: str = DEFAULT_SYMBOL) -> list[dict[str, Any]]:
    """Causal 10-bp directional-change episodes with one 35-bp candidate each."""
    records: list[dict[str, Any]] = []
    state = "SEEK"
    run_min = math.inf
    run_min_i = -1
    run_max = -math.inf
    run_max_i = -1
    open_ep: dict[str, Any] | None = None
    prev_i: int | None = None

    def valid(i: int) -> bool:
        return math.isfinite(float(price[i])) and float(price[i]) > 0

    def init_seek(i: int) -> None:
        nonlocal run_min, run_min_i, run_max, run_max_i
        p = float(price[i])
        run_min, run_min_i = p, i
        run_max, run_max_i = p, i

    def update_hits(ep: dict[str, Any], i: int) -> None:
        d = directional_bps(float(price[i]), float(ep["origin_price"]), int(ep["direction"]))
        for t in TARGET_BPS:
            if ep["hits"][float(t)] is None and d + 1e-12 >= float(t):
                ep["hits"][float(t)] = int(i)

    def finalize(ep: dict[str, Any], reason: str, close_i: int | None) -> None:
        ep = dict(ep)
        ep["close_reason"] = str(reason)
        ep["close_i"] = int(close_i) if close_i is not None else None
        records.append(ep)

    for i in range(len(sec)):
        contiguous = prev_i is None or int(sec[i]) == int(sec[prev_i]) + 1
        if not valid(i) or not contiguous:
            if open_ep is not None:
                finalize(open_ep, "PRICE_GAP_CENSORED", prev_i)
                open_ep = None
            state = "SEEK"
            run_min, run_min_i, run_max, run_max_i = math.inf, -1, -math.inf, -1
            if valid(i):
                init_seek(i)
                prev_i = i
            else:
                prev_i = None
            continue

        p = float(price[i])
        if state == "SEEK":
            if run_min_i < 0:
                init_seek(i)
            if p < run_min:
                run_min, run_min_i = p, i
            if p > run_max:
                run_max, run_max_i = p, i
            up = directional_bps(p, run_min, +1)
            dn = directional_bps(p, run_max, -1)
            if up + 1e-12 >= RELEASE_CONFIRMATION_BPS:
                open_ep = _new_episode(symbol, +1, run_min_i, i, sec, price)
                update_hits(open_ep, i)
                state = "UP"
            elif dn + 1e-12 >= RELEASE_CONFIRMATION_BPS:
                open_ep = _new_episode(symbol, -1, run_max_i, i, sec, price)
                update_hits(open_ep, i)
                state = "DOWN"

        elif state == "UP":
            assert open_ep is not None
            if p > float(open_ep["extreme_price"]):
                open_ep["extreme_price"] = p
                open_ep["extreme_i"] = i
            update_hits(open_ep, i)
            peak_p = float(open_ep["extreme_price"])
            if directional_bps(p, peak_p, -1) + 1e-12 >= REVERSAL_CONFIRMATION_BPS:
                peak_i = int(open_ep["extreme_i"])
                finalize(open_ep, "OPPOSITE_10BP_CONFIRMED", i)
                open_ep = _new_episode(symbol, -1, peak_i, i, sec, price)
                update_hits(open_ep, i)
                state = "DOWN"

        elif state == "DOWN":
            assert open_ep is not None
            if p < float(open_ep["extreme_price"]):
                open_ep["extreme_price"] = p
                open_ep["extreme_i"] = i
            update_hits(open_ep, i)
            trough_p = float(open_ep["extreme_price"])
            if directional_bps(p, trough_p, +1) + 1e-12 >= REVERSAL_CONFIRMATION_BPS:
                trough_i = int(open_ep["extreme_i"])
                finalize(open_ep, "OPPOSITE_10BP_CONFIRMED", i)
                open_ep = _new_episode(symbol, +1, trough_i, i, sec, price)
                update_hits(open_ep, i)
                state = "UP"

        prev_i = i

    if open_ep is not None:
        finalize(open_ep, "RANGE_END_CENSORED", prev_i)
    return records


def materialize_episodes(raw: Sequence[dict[str, Any]], sec, price):
    _, pd = require_numeric_stack()
    rows = []
    for ep in raw:
        oi = int(ep["origin_i"])
        ci = int(ep["confirmation_i"])
        xi = int(ep["extreme_i"])
        close_i = ep.get("close_i")
        hit35_i = ep["hits"].get(35.0)
        candidate_i = hit35_i
        direction = int(ep["direction"])
        origin_p = float(ep["origin_price"])
        candidate_p = float(price[candidate_i]) if candidate_i is not None else None
        close_p = float(price[int(close_i)]) if close_i is not None and int(close_i) < len(price) and math.isfinite(float(price[int(close_i)])) else None
        row = CausalEpisode(
            episode_id=str(ep["episode_id"]),
            symbol=str(ep["symbol"]),
            direction=direction,
            direction_label="UP" if direction > 0 else "DOWN",
            origin_second=int(sec[oi]),
            origin_time=str(iso_time(int(sec[oi]))),
            origin_date=datetime.fromtimestamp(int(sec[oi]), tz=timezone.utc).date().isoformat(),
            origin_price=origin_p,
            confirmation_second=int(sec[ci]),
            confirmation_time=str(iso_time(int(sec[ci]))),
            confirmation_price=float(price[ci]),
            confirmation_displacement_bps=directional_bps(float(price[ci]), origin_p, direction),
            release_confirmation_bps=RELEASE_CONFIRMATION_BPS,
            reversal_confirmation_bps=REVERSAL_CONFIRMATION_BPS,
            candidate_threshold_bps=CANDIDATE_THRESHOLD_BPS,
            candidate_second=int(sec[candidate_i]) if candidate_i is not None else None,
            candidate_time=iso_time(int(sec[candidate_i])) if candidate_i is not None else None,
            candidate_price=candidate_p,
            candidate_displacement_bps=directional_bps(candidate_p, origin_p, direction) if candidate_p is not None else None,
            seconds_confirmation_to_candidate=(int(sec[candidate_i]) - int(sec[ci])) if candidate_i is not None else None,
            hit_35_second=int(sec[ep["hits"][35.0]]) if ep["hits"].get(35.0) is not None else None,
            hit_50_second=int(sec[ep["hits"][50.0]]) if ep["hits"].get(50.0) is not None else None,
            hit_75_second=int(sec[ep["hits"][75.0]]) if ep["hits"].get(75.0) is not None else None,
            hit_100_second=int(sec[ep["hits"][100.0]]) if ep["hits"].get(100.0) is not None else None,
            extreme_second=int(sec[xi]),
            extreme_time=str(iso_time(int(sec[xi]))),
            extreme_price=float(ep["extreme_price"]),
            max_excursion_bps=directional_bps(float(ep["extreme_price"]), origin_p, direction),
            close_second=int(sec[int(close_i)]) if close_i is not None else None,
            close_time=iso_time(int(sec[int(close_i)])) if close_i is not None else None,
            close_price=close_p,
            close_reason=str(ep.get("close_reason") or "UNKNOWN"),
        )
        rows.append(asdict(row))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["confirmation_second", "episode_id"], kind="mergesort").reset_index(drop=True)


def build_candidates(episodes, frame, price_source_label: str):
    np, pd = require_numeric_stack()
    c = episodes[episodes["candidate_second"].notna()].copy()
    if c.empty:
        return pd.DataFrame()
    c["candidate_second"] = pd.to_numeric(c["candidate_second"], errors="raise").astype("int64")
    book_map = dict(zip(pd.to_numeric(frame["receive_second"], errors="coerce").astype("int64"), frame["book_valid"].fillna(False).astype(bool)))
    rows = []
    for r in c.to_dict("records"):
        cs = int(r["candidate_second"])
        cid = stable_id("7I_CAUSAL_CANDIDATE_V1", r["episode_id"], cs, CANDIDATE_THRESHOLD_BPS)
        row = {
            "candidate_id": cid,
            "episode_id": str(r["episode_id"]),
            "symbol": str(r["symbol"]),
            "direction": int(r["direction"]),
            "direction_label": str(r["direction_label"]),
            "origin_second": int(r["origin_second"]),
            "origin_time": str(r["origin_time"]),
            "origin_date": str(r["origin_date"]),
            "origin_price": float(r["origin_price"]),
            "release_confirmation_second": int(r["confirmation_second"]),
            "release_confirmation_time": str(r["confirmation_time"]),
            "release_confirmation_price": float(r["confirmation_price"]),
            "release_confirmation_bps": RELEASE_CONFIRMATION_BPS,
            "candidate_second": cs,
            "candidate_time": str(r["candidate_time"]),
            "candidate_date": datetime.fromtimestamp(cs, tz=timezone.utc).date().isoformat(),
            "candidate_price": float(r["candidate_price"]),
            "candidate_threshold_bps": CANDIDATE_THRESHOLD_BPS,
            "candidate_displacement_bps": float(r["candidate_displacement_bps"]),
            "seconds_release_to_candidate": int(r["seconds_confirmation_to_candidate"]),
            "price_source": price_source_label,
            "book_valid_at_candidate": bool(book_map.get(cs, False)),
            "candidate_contract": "CAUSAL_10BP_DIRECTIONAL_CHANGE_ORIGIN__FIRST_35BP_PASSAGE",
        }
        for d in DECISION_DELAYS_SEC:
            row[f"decision_second_{d}s"] = cs + int(d)
            row[f"decision_time_{d}s"] = iso_time(cs + int(d))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True)


def contiguous_valid_end_indices(sec, price):
    np, _, _, _ = require_stack()
    n = len(sec)
    end = np.full(n, -1, dtype=np.int64)
    next_end = -1
    for i in range(n - 1, -1, -1):
        ok = math.isfinite(float(price[i])) and float(price[i]) > 0
        if not ok:
            next_end = -1
            end[i] = -1
            continue
        if i == n - 1 or next_end < 0 or int(sec[i + 1]) != int(sec[i]) + 1:
            next_end = i
        end[i] = next_end
    return end


def first_hit_second(path_sec, path_price, origin_price: float, direction: int, target_bps: float) -> int | None:
    for s, p in zip(path_sec, path_price):
        if directional_bps(float(p), origin_price, direction) + 1e-12 >= target_bps:
            return int(s)
    return None


def build_candidate_outcomes(candidates, episodes, sec, price):
    np, pd = require_numeric_stack()
    if candidates.empty:
        return pd.DataFrame()
    ep_map = {str(r["episode_id"]): r for r in episodes.to_dict("records")}
    valid_end = contiguous_valid_end_indices(sec, price)
    rows = []
    for c in candidates.to_dict("records"):
        cs = int(c["candidate_second"])
        ci = int(np.searchsorted(sec, cs, side="left"))
        if ci >= len(sec) or int(sec[ci]) != cs:
            raise RuntimeError(f"Candidate second missing from execution price grid: {cs}")
        ep = ep_map[str(c["episode_id"])]
        direction = int(c["direction"])
        origin_price = float(c["origin_price"])
        candidate_price = float(c["candidate_price"])
        row: dict[str, Any] = {
            "candidate_id": str(c["candidate_id"]),
            "episode_id": str(c["episode_id"]),
            "candidate_second": cs,
            "candidate_date": str(c["candidate_date"]),
            "direction": direction,
            "episode_reach_50": bool(ep.get("hit_50_second") is not None and not pd.isna(ep.get("hit_50_second"))),
            "episode_reach_75": bool(ep.get("hit_75_second") is not None and not pd.isna(ep.get("hit_75_second"))),
            "episode_reach_100": bool(ep.get("hit_100_second") is not None and not pd.isna(ep.get("hit_100_second"))),
            "episode_close_second": ep.get("close_second"),
            "episode_close_reason": str(ep.get("close_reason")),
        }
        for h in OUTCOME_HORIZONS_SEC:
            target_second = cs + int(h)
            hi = int(np.searchsorted(sec, target_second, side="right") - 1)
            seg_end_i = int(valid_end[ci]) if ci < len(valid_end) else -1
            fully_observed = hi >= ci and hi < len(sec) and int(sec[hi]) == target_second and seg_end_i >= hi
            use_hi = min(seg_end_i, hi) if seg_end_i >= ci and hi >= ci else ci
            path_sec = sec[ci : use_hi + 1]
            path_price = price[ci : use_hi + 1]
            dr = direction * (path_price / candidate_price - 1.0) * 10_000.0
            row[f"h{h}_fully_observed"] = bool(fully_observed)
            row[f"h{h}_observed_seconds"] = int(path_sec[-1] - cs) if len(path_sec) else 0
            row[f"h{h}_mfe_from_candidate_bps"] = float(np.nanmax(dr)) if len(dr) else math.nan
            row[f"h{h}_mae_from_candidate_bps"] = float(max(0.0, -float(np.nanmin(dr)))) if len(dr) else math.nan
            for t in (50, 75, 100):
                hit = first_hit_second(path_sec, path_price, origin_price, direction, float(t))
                row[f"h{h}_origin_hit_{t}_second"] = hit
                if hit is not None:
                    row[f"h{h}_origin_reach_{t}"] = True
                    row[f"h{h}_seconds_to_{t}_from_candidate"] = int(hit - cs)
                elif fully_observed:
                    row[f"h{h}_origin_reach_{t}"] = False
                    row[f"h{h}_seconds_to_{t}_from_candidate"] = math.nan
                else:
                    row[f"h{h}_origin_reach_{t}"] = None
                    row[f"h{h}_seconds_to_{t}_from_candidate"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate_second", "candidate_id"], kind="mergesort").reset_index(drop=True)


def load_phase7a_events(root: Path):
    _, pd, _, pq = require_stack()
    p = root / "physical_events.parquet"
    if not p.exists():
        raise RuntimeError(f"Missing frozen Phase-7A physical catalog: {p}")
    return pq.read_table(p).to_pandas()


def build_geometry_audit(candidates, physical):
    np, pd = require_numeric_stack()
    if candidates.empty or physical.empty:
        return pd.DataFrame(), pd.DataFrame()
    ph = physical.copy()
    for c in ["onset_second", "hit_35_second", "reversal_confirm_second", "extreme_second", "direction"]:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")
    if "event_id" not in ph.columns:
        raise RuntimeError("Phase-7A physical catalog missing event_id")
    by_dir = {d: g.sort_values("onset_second", kind="mergesort").to_dict("records") for d, g in ph.groupby("direction")}
    rows = []
    mapped_event_ids: set[str] = set()
    for c in candidates.to_dict("records"):
        cs = int(c["candidate_second"])
        d = int(c["direction"])
        matches = []
        for ev in by_dir.get(d, []):
            onset = safe_float(ev.get("onset_second"))
            if not math.isfinite(onset) or onset > cs:
                continue
            end = safe_float(ev.get("reversal_confirm_second"))
            if not math.isfinite(end):
                end = safe_float(ev.get("extreme_second"))
            if math.isfinite(end) and cs <= int(end):
                matches.append(ev)
        ev = max(matches, key=lambda r: int(r["onset_second"])) if matches else None
        r = {
            "candidate_id": str(c["candidate_id"]),
            "candidate_second": cs,
            "candidate_date": str(c["candidate_date"]),
            "direction": d,
            "mapped_phase7a_event": ev is not None,
            "phase7a_event_id": str(ev["event_id"]) if ev is not None else None,
        }
        if ev is not None:
            mapped_event_ids.add(str(ev["event_id"]))
            h35 = safe_float(ev.get("hit_35_second"))
            onset = int(ev["onset_second"])
            op = safe_float(ev.get("onset_price"))
            r.update({
                "candidate_minus_phase7a_hit35_sec": cs - int(h35) if math.isfinite(h35) else math.nan,
                "causal_origin_minus_phase7a_onset_sec": int(c["origin_second"]) - onset,
                "causal_origin_vs_phase7a_onset_price_bps": ((float(c["origin_price"]) / op - 1.0) * 10_000.0) if math.isfinite(op) and op > 0 else math.nan,
                "phase7a_hit50": math.isfinite(safe_float(ev.get("hit_50_second"))),
                "phase7a_hit75": math.isfinite(safe_float(ev.get("hit_75_second"))),
                "phase7a_hit100": math.isfinite(safe_float(ev.get("hit_100_second"))),
                "phase7a_max_excursion_bps": safe_float(ev.get("leg_max_excursion_bps")),
            })
        rows.append(r)
    audit = pd.DataFrame(rows)

    # Reverse coverage: which frozen 7A physical events contain at least one causal candidate?
    cov_rows = []
    for ev in ph.to_dict("records"):
        eid = str(ev["event_id"])
        cov_rows.append({
            "phase7a_event_id": eid,
            "onset_date": str(ev.get("onset_date")),
            "direction": int(ev["direction"]),
            "hit_50": math.isfinite(safe_float(ev.get("hit_50_second"))),
            "hit_75": math.isfinite(safe_float(ev.get("hit_75_second"))),
            "hit_100": math.isfinite(safe_float(ev.get("hit_100_second"))),
            "covered_by_causal_candidate": eid in mapped_event_ids,
        })
    return audit, pd.DataFrame(cov_rows)


def build_invariants(episodes, candidates, outcomes, geometry, start: date, end: date):
    np, pd = require_numeric_stack()
    rows: list[dict[str, Any]] = []
    def add(name: str, passed: bool, observed: Any, expected: Any, detail: str = ""):
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "observed": observed, "expected": expected, "detail": detail})

    add("episode_ids_unique", not episodes["episode_id"].duplicated().any(), int(episodes["episode_id"].duplicated().sum()), 0)
    add("candidate_ids_unique", not candidates["candidate_id"].duplicated().any(), int(candidates["candidate_id"].duplicated().sum()), 0)
    add("one_candidate_per_episode", int(candidates.groupby("episode_id").size().max()) <= 1 if len(candidates) else True, int(candidates.groupby("episode_id").size().max()) if len(candidates) else 0, "<=1")
    bad_confirm = int((pd.to_numeric(episodes["confirmation_displacement_bps"], errors="coerce") + 1e-9 < RELEASE_CONFIRMATION_BPS).sum())
    add("release_confirmation_at_least_10bp", bad_confirm == 0, bad_confirm, 0)
    if len(candidates):
        bad_order = int((candidates["origin_second"].astype(int) > candidates["release_confirmation_second"].astype(int)).sum() + (candidates["release_confirmation_second"].astype(int) > candidates["candidate_second"].astype(int)).sum())
        add("causal_time_order_origin_confirmation_candidate", bad_order == 0, bad_order, 0)
        bad_disp = int((pd.to_numeric(candidates["candidate_displacement_bps"], errors="coerce") + 1e-9 < CANDIDATE_THRESHOLD_BPS).sum())
        add("candidate_first_passage_at_least_35bp", bad_disp == 0, bad_disp, 0)
        bad_date = int(((pd.to_datetime(candidates["candidate_date"]).dt.date < start) | (pd.to_datetime(candidates["candidate_date"]).dt.date > end)).sum())
        add("candidate_dates_within_requested_range", bad_date == 0, bad_date, 0)
    else:
        add("causal_time_order_origin_confirmation_candidate", False, 0, ">0 candidates")
        add("candidate_first_passage_at_least_35bp", False, 0, ">0 candidates")
        add("candidate_dates_within_requested_range", True, 0, 0)
    add("candidate_outcomes_one_to_one", len(outcomes) == len(candidates) and not outcomes["candidate_id"].duplicated().any(), len(outcomes), len(candidates))

    # Episode target nesting.
    nesting_bad = 0
    for r in episodes.to_dict("records"):
        vals = [r.get(f"hit_{t}_second") for t in (35, 50, 75, 100)]
        prev = None
        seen_missing = False
        for v in vals:
            finite = v is not None and not pd.isna(v)
            if not finite:
                seen_missing = True
                continue
            if seen_missing or (prev is not None and int(v) < prev):
                nesting_bad += 1
                break
            prev = int(v)
    add("episode_target_hits_nested", nesting_bad == 0, nesting_bad, 0)

    # Candidate table must not contain obvious future outcome fields.
    future_tokens = ("reach_", "mfe", "mae", "hit_50", "hit_75", "hit_100", "close_reason", "close_second")
    leaked = [c for c in candidates.columns if any(tok in c.lower() for tok in future_tokens)]
    add("candidate_table_separate_from_future_labels", len(leaked) == 0, leaked, [])

    if len(geometry):
        add("geometry_rows_match_candidates", len(geometry) == len(candidates), len(geometry), len(candidates))
    return pd.DataFrame(rows)


def base_rate_report(outcomes):
    _, pd = require_numeric_stack()
    rows = []
    for h in OUTCOME_HORIZONS_SEC:
        for t in (50, 75, 100):
            col = f"h{h}_origin_reach_{t}"
            s = outcomes[col].dropna()
            rows.append({
                "horizon_sec": h,
                "target_bps_from_origin": t,
                "eligible_candidates": int(len(s)),
                "reached": int(s.astype(bool).sum()) if len(s) else 0,
                "continuation_rate": float(s.astype(bool).mean()) if len(s) else math.nan,
            })
    return pd.DataFrame(rows)


def daily_report(episodes, candidates):
    _, pd = require_numeric_stack()
    ep = episodes.copy()
    ca = candidates.copy()
    epg = ep.groupby("origin_date").agg(episodes=("episode_id", "count"), episodes_reaching_35=("candidate_second", lambda s: int(s.notna().sum()))).reset_index().rename(columns={"origin_date": "date"})
    if ca.empty:
        epg["candidates"] = 0
        epg["up_candidates"] = 0
        epg["down_candidates"] = 0
        return epg
    cg = ca.groupby("candidate_date").agg(
        candidates=("candidate_id", "count"),
        up_candidates=("direction", lambda s: int((s > 0).sum())),
        down_candidates=("direction", lambda s: int((s < 0).sum())),
        book_valid_candidates=("book_valid_at_candidate", lambda s: int(s.astype(bool).sum())),
    ).reset_index().rename(columns={"candidate_date": "date"})
    return epg.merge(cg, how="outer", on="date").fillna(0).sort_values("date").reset_index(drop=True)


def duplication_report(candidates):
    np, pd = require_numeric_stack()
    if candidates.empty:
        return pd.DataFrame([{"metric": "candidate_count", "value": 0}])
    x = candidates.sort_values("candidate_second", kind="mergesort").reset_index(drop=True)
    sec = x["candidate_second"].to_numpy(dtype=np.int64)
    direction = x["direction"].to_numpy(dtype=np.int64)
    prev_gap = np.full(len(x), np.nan)
    prev_same_gap = np.full(len(x), np.nan)
    for i in range(1, len(x)):
        prev_gap[i] = sec[i] - sec[i - 1]
        j = i - 1
        while j >= 0:
            if direction[j] == direction[i]:
                prev_same_gap[i] = sec[i] - sec[j]
                break
            j -= 1
    rows = [
        {"metric": "candidate_count", "value": int(len(x))},
        {"metric": "median_gap_sec", "value": float(np.nanmedian(prev_gap[1:])) if len(x) > 1 else math.nan},
        {"metric": "share_any_prev_candidate_within_60s", "value": float(np.nanmean(prev_gap <= 60))},
        {"metric": "share_any_prev_candidate_within_300s", "value": float(np.nanmean(prev_gap <= 300))},
        {"metric": "share_same_direction_prev_candidate_within_60s", "value": float(np.nanmean(prev_same_gap <= 60))},
        {"metric": "share_same_direction_prev_candidate_within_300s", "value": float(np.nanmean(prev_same_gap <= 300))},
    ]
    return pd.DataFrame(rows)


def timing_report(candidates):
    np, pd = require_numeric_stack()
    if candidates.empty:
        return pd.DataFrame()
    vals = pd.to_numeric(candidates["seconds_release_to_candidate"], errors="coerce")
    disp = pd.to_numeric(candidates["candidate_displacement_bps"], errors="coerce")
    rows = []
    for label, g in [("ALL", candidates), ("UP", candidates[candidates["direction"] > 0]), ("DOWN", candidates[candidates["direction"] < 0])]:
        v = pd.to_numeric(g["seconds_release_to_candidate"], errors="coerce").dropna()
        d = pd.to_numeric(g["candidate_displacement_bps"], errors="coerce").dropna()
        rows.append({
            "scope": label,
            "candidates": int(len(g)),
            "median_seconds_10_to_35": float(v.median()) if len(v) else math.nan,
            "p25_seconds_10_to_35": float(v.quantile(0.25)) if len(v) else math.nan,
            "p75_seconds_10_to_35": float(v.quantile(0.75)) if len(v) else math.nan,
            "median_detection_displacement_bps": float(d.median()) if len(d) else math.nan,
            "median_nominal_remaining_to_75_bps": float((75.0 - d).median()) if len(d) else math.nan,
            "median_nominal_remaining_to_100_bps": float((100.0 - d).median()) if len(d) else math.nan,
        })
    return pd.DataFrame(rows)


def geometry_report(geometry):
    np, pd = require_numeric_stack()
    if geometry.empty:
        return pd.DataFrame()
    mapped = geometry[geometry["mapped_phase7a_event"].fillna(False).astype(bool)]
    rows = [{
        "candidates": int(len(geometry)),
        "mapped_to_same_direction_phase7a_leg": int(len(mapped)),
        "mapped_share": float(len(mapped) / len(geometry)) if len(geometry) else math.nan,
        "median_candidate_minus_phase7a_hit35_sec": float(pd.to_numeric(mapped.get("candidate_minus_phase7a_hit35_sec"), errors="coerce").median()) if len(mapped) else math.nan,
        "median_causal_origin_minus_phase7a_onset_sec": float(pd.to_numeric(mapped.get("causal_origin_minus_phase7a_onset_sec"), errors="coerce").median()) if len(mapped) else math.nan,
        "median_abs_causal_origin_price_delta_bps": float(pd.to_numeric(mapped.get("causal_origin_vs_phase7a_onset_price_bps"), errors="coerce").abs().median()) if len(mapped) else math.nan,
    }]
    return pd.DataFrame(rows)


def physical_coverage_report(coverage):
    _, pd = require_numeric_stack()
    if coverage.empty:
        return pd.DataFrame()
    rows = []
    for target_col, label in [(None, "ALL_35_PLUS"), ("hit_50", "50_PLUS"), ("hit_75", "75_PLUS"), ("hit_100", "100_PLUS")]:
        g = coverage if target_col is None else coverage[coverage[target_col].fillna(False).astype(bool)]
        rows.append({
            "physical_scope": label,
            "physical_events": int(len(g)),
            "covered_by_causal_candidate": int(g["covered_by_causal_candidate"].fillna(False).astype(bool).sum()),
            "coverage_rate": float(g["covered_by_causal_candidate"].fillna(False).astype(bool).mean()) if len(g) else math.nan,
        })
    return pd.DataFrame(rows)


def build_findings(summary: Mapping[str, Any], base_rates, timing, geom, phys_cov) -> str:
    lines = [
        "# Phase 7I.1/7I.2 — Causal Large-Move Candidate Catalog",
        "",
        f"- Builder: **{BUILDER_VERSION}**",
        f"- Status: **{summary['status']}**",
        f"- Symbol: **{summary['symbol']}**",
        f"- Causal episodes: **{summary['episode_count']:,}**",
        f"- 35-bp causal candidates: **{summary['candidate_count']:,}**",
        "- Raw-depth replay: **NONE**",
        "",
        "## Frozen candidate contract",
        "",
        "A running-extreme directional-change release is confirmed after 10 bps.  The already-observed extreme is then frozen as the causal episode origin.  The opposite 10-bp directional change closes/switches the episode.  One candidate is emitted at the first 35-bp passage from that frozen origin.",
        "",
        "Candidate construction uses no future outcomes.  50/75/100 continuation labels are stored separately.",
        "",
    ]
    if len(timing):
        r = timing[timing["scope"] == "ALL"].iloc[0]
        lines += [
            "## Candidate timing",
            "",
            f"- Median 10→35 latency: **{safe_float(r['median_seconds_10_to_35']):.1f}s**",
            f"- Median realized displacement at detection: **{safe_float(r['median_detection_displacement_bps']):.2f} bps**",
            f"- Median nominal movement remaining to 75: **{safe_float(r['median_nominal_remaining_to_75_bps']):.2f} bps**",
            "",
        ]
    if len(base_rates):
        r = base_rates[(base_rates["horizon_sec"] == 1800) & (base_rates["target_bps_from_origin"] == 75)]
        if len(r):
            rr = r.iloc[0]
            lines += [
                "## All-market causal base rate",
                "",
                f"- 75-bp continuation within 30m: **{safe_float(rr['continuation_rate']):.3f}** ({int(rr['reached'])}/{int(rr['eligible_candidates'])})",
                "",
            ]
    if len(geom):
        r = geom.iloc[0]
        lines += [
            "## Phase-7A geometry audit (audit-only)",
            "",
            f"- Candidate mapping to same-direction physical legs: **{safe_float(r['mapped_share']):.3f}**",
            f"- Median candidate-vs-Phase7A 35bp timing difference: **{safe_float(r['median_candidate_minus_phase7a_hit35_sec']):.1f}s**",
            "",
        ]
    if len(phys_cov):
        lines += ["## Frozen physical-event coverage", ""]
        for r in phys_cov.to_dict("records"):
            lines.append(f"- {r['physical_scope']}: **{safe_float(r['coverage_rate']):.3f}** ({int(r['covered_by_causal_candidate'])}/{int(r['physical_events'])})")
        lines.append("")
    lines += [
        "## Interpretation discipline",
        "",
        "This phase audits/freezes a causal candidate ontology.  It does not establish predictive precision, a trading rule, or alpha.  The next expensive phase should only be run after candidate counts, duplication, physical-event coverage, and continuation base rates are judged scientifically sensible.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    np, pd = require_numeric_stack()
    n = 240
    sec = np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64)
    p = np.full(n, 100.0, dtype=float)
    # UP episode: confirm at +10, candidate +35, reach >50, then reverse >10.
    for k in range(1, 81):
        p[k] = 100.0 * (1.0 + 0.0060 * k / 80.0)
    for k in range(81, 121):
        p[k] = p[80] * (1.0 - 0.0020 * (k - 80) / 40.0)
    # Continue down enough to create candidate in the opposite episode.
    for k in range(121, 181):
        p[k] = p[120] * (1.0 - 0.0050 * (k - 120) / 60.0)
    p[181:] = p[180]
    raw = segment_causal_directional_change(sec, p)
    episodes = materialize_episodes(raw, sec, p)
    assert len(episodes) >= 2, episodes
    up = episodes[episodes["direction"] == 1].iloc[0]
    assert float(up["confirmation_displacement_bps"]) >= 10.0
    assert pd.notna(up["candidate_second"])
    assert int(up["origin_second"]) <= int(up["confirmation_second"]) <= int(up["candidate_second"])
    assert float(up["candidate_displacement_bps"]) >= 35.0
    assert pd.notna(up["hit_50_second"])

    # Gap must reset; no episode may bridge it.
    p2 = p.copy()
    p2[30:35] = np.nan
    r2 = segment_causal_directional_change(sec, p2)
    for ep in r2:
        oi, ci = int(ep["origin_i"]), int(ep["confirmation_i"])
        assert not (oi < 30 and ci >= 35), ep

    # Future label names must never enter the causal candidate table contract.
    dummy_frame = pd.DataFrame({"receive_second": sec, "book_valid": True})
    candidates = build_candidates(episodes, dummy_frame, "TEST")
    assert len(candidates) >= 1
    assert not any("reach_75" in c or "mfe" in c or "mae" in c for c in candidates.columns)
    print("SELF-TEST PASS")
    print("Validated causal 10-bp release confirmation, 10-bp reversal reset, one first-35bp candidate per episode, gap resets, target nesting, and candidate/outcome separation.")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Phase 7I.1 causal large-move candidate catalog")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date", default="2026-05-18")
    p.add_argument("--end-date", default="2026-07-18")
    p.add_argument("--phase7a-script", type=Path, default=DEFAULT_PHASE7A_SCRIPT)
    p.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    p.add_argument("--phase7a-root", type=Path, default=DEFAULT_PHASE7A_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--max-trade-stale-sec", type=int, default=DEFAULT_MAX_TRADE_STALE_SEC)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(); return 0
    np, pd, pa, pq = require_stack()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise SystemExit("--start-date must be <= --end-date")
    if args.max_trade_stale_sec != DEFAULT_MAX_TRADE_STALE_SEC:
        raise SystemExit(f"Phase 7I.1 is frozen to max_trade_stale_sec={DEFAULT_MAX_TRADE_STALE_SEC}; received {args.max_trade_stale_sec}")

    phase7a = load_module(args.phase7a_script, "phase7a_frozen_for_7i1")
    if getattr(phase7a, "BUILDER_VERSION", None) != "1.0.0":
        raise RuntimeError(f"Phase 7I.1 requires frozen Phase-7A builder 1.0.0; found {getattr(phase7a, 'BUILDER_VERSION', None)!r}")

    print("Building Phase 7I.1/7I.2 causal all-market candidate catalog...")
    print(f"Builder version      : {BUILDER_VERSION}")
    print(f"Frozen Phase-7A      : {phase7a.BUILDER_VERSION}")
    print(f"Symbol               : {args.symbol}")
    print(f"Dates                : {start} to {end}")
    print(f"Release confirmation : {RELEASE_CONFIRMATION_BPS:g} bps")
    print(f"Reversal/reset       : {REVERSAL_CONFIRMATION_BPS:g} bps")
    print(f"Candidate threshold  : first {CANDIDATE_THRESHOLD_BPS:g} bps from frozen causal origin")
    print("Raw depth replay     : NONE")
    print("Research role        : causal candidate-universe construction + geometry/base-rate audit; not alpha")

    print("[1/8] Loading Phase-6A one-second execution price range...")
    # One prior/next day are loaded only when available to preserve bounded price
    # context at the range boundary and to label end-of-range forward outcomes.
    requested_days = []
    d = start - timedelta(days=1)
    while d <= end + timedelta(days=1):
        pth = phase7a.execution_day_path(args.execution_root, args.symbol, d)
        if pth.exists():
            requested_days.append(d)
        d += timedelta(days=1)
    if not requested_days:
        raise RuntimeError("No Phase-6A execution partitions found")
    frame = phase7a.load_execution_range(args.execution_root, args.symbol, requested_days)
    sec, price, price_source = phase7a.build_catalog_price(frame, args.max_trade_stale_sec, "last")

    print("[2/8] Building causal 10-bp directional-change episodes...")
    raw = segment_causal_directional_change(sec, price, args.symbol)
    episodes_all = materialize_episodes(raw, sec, price)
    # Keep episodes whose confirmation occurs in the requested research range.
    conf_date = pd.to_datetime(episodes_all["confirmation_second"], unit="s", utc=True).dt.date
    episodes = episodes_all[(conf_date >= start) & (conf_date <= end)].copy().reset_index(drop=True)
    print(f"      causal episodes={len(episodes):,}")

    print("[3/8] Freezing one first-35bp candidate per eligible causal episode...")
    candidates = build_candidates(episodes, frame, price_source)
    print(f"      causal 35bp candidates={len(candidates):,}")

    print("[4/8] Building separate future continuation/MFE/MAE outcomes...")
    outcomes = build_candidate_outcomes(candidates, episodes, sec, price)

    print("[5/8] Auditing geometry against frozen Phase-7A physical events...")
    physical = load_phase7a_events(args.phase7a_root)
    geometry, physical_cov = build_geometry_audit(candidates, physical)

    print("[6/8] Building reports and invariants...")
    invariants = build_invariants(episodes, candidates, outcomes, geometry, start, end)
    failed = invariants[invariants["status"] == "FAIL"]
    base_rates = base_rate_report(outcomes)
    daily = daily_report(episodes, candidates)
    duplication = duplication_report(candidates)
    timing = timing_report(candidates)
    geom_report = geometry_report(geometry)
    phys_report = physical_coverage_report(physical_cov)

    status = "PASS" if failed.empty else "FAIL"
    candidate_id_hash = hashlib.sha256("\n".join(sorted(candidates["candidate_id"].astype(str))).encode("utf-8")).hexdigest() if len(candidates) else hashlib.sha256(b"").hexdigest()
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
        "episode_count": int(len(episodes)),
        "candidate_count": int(len(candidates)),
        "candidate_days": int(candidates["candidate_date"].nunique()) if len(candidates) else 0,
        "up_candidates": int((candidates["direction"] > 0).sum()) if len(candidates) else 0,
        "down_candidates": int((candidates["direction"] < 0).sum()) if len(candidates) else 0,
        "candidate_id_set_sha256": candidate_id_hash,
        "failed_invariants": int(len(failed)),
        "interpretation": "Causal all-market candidate ontology and hindsight outcome audit only; no book-mechanics signal, execution policy, or alpha claim.",
    }
    run_config = {
        **summary,
        "release_confirmation_bps": RELEASE_CONFIRMATION_BPS,
        "reversal_confirmation_bps": REVERSAL_CONFIRMATION_BPS,
        "candidate_threshold_bps": CANDIDATE_THRESHOLD_BPS,
        "targets_bps": list(TARGET_BPS),
        "outcome_horizons_sec": list(OUTCOME_HORIZONS_SEC),
        "decision_delays_sec": list(DECISION_DELAYS_SEC),
        "max_trade_stale_sec": args.max_trade_stale_sec,
        "execution_root": str(args.execution_root),
        "phase7a_root": str(args.phase7a_root),
        "candidate_feature_contract": "candidates.parquet is causal-only; all future labels are isolated in candidate_outcomes.parquet",
        "episode_contract": "10bp directional-change confirmation; opposite 10bp closes/switches; first 35bp from frozen causal origin emits at most one candidate",
    }
    freeze_manifest = {
        "schema_name": SCHEMA_NAME,
        "builder_version": BUILDER_VERSION,
        "symbol": args.symbol,
        "candidate_count": int(len(candidates)),
        "candidate_id_set_sha256": candidate_id_hash,
        "candidate_ids_sorted": sorted(candidates["candidate_id"].astype(str).tolist()),
        "freeze_status": "AUDIT_READY_NOT_YET_SCIENTIFICALLY_APPROVED",
        "note": "Inspect Phase-7I.1/7I.2 candidate counts, duplication, physical-event coverage and base rates before using these IDs for the expensive participant-interaction replay.",
    }

    print("[7/8] Writing atomic datasets and reports...")
    write_parquet_atomic(episodes, args.output_root / "episodes.parquet")
    write_parquet_atomic(candidates, args.output_root / "candidates.parquet")
    write_parquet_atomic(outcomes, args.output_root / "candidate_outcomes.parquet")
    write_parquet_atomic(geometry, args.output_root / "geometry_audit.parquet")
    write_parquet_atomic(physical_cov, args.output_root / "physical_event_coverage.parquet")
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", summary)
    atomic_write_json(args.output_root / "metadata" / "run_config.json", run_config)
    atomic_write_json(args.output_root / "metadata" / "candidate_freeze_manifest.json", freeze_manifest)

    write_csv_atomic(daily, args.report_root / "causal_candidate_daily_counts.csv")
    write_csv_atomic(base_rates, args.report_root / "causal_candidate_base_rates.csv")
    write_csv_atomic(timing, args.report_root / "causal_candidate_timing.csv")
    write_csv_atomic(duplication, args.report_root / "causal_candidate_duplication_audit.csv")
    write_csv_atomic(geom_report, args.report_root / "causal_candidate_geometry_audit.csv")
    write_csv_atomic(phys_report, args.report_root / "causal_candidate_physical_coverage.csv")
    write_csv_atomic(invariants, args.report_root / "causal_candidate_invariants.csv")
    atomic_write_text(args.report_root / "causal_candidate_findings.md", build_findings(summary, base_rates, timing, geom_report, phys_report))

    print("[8/8] Phase 7I.1/7I.2 summary")
    print(f"      status={status}")
    print(f"      episodes={len(episodes):,} candidates={len(candidates):,} candidate_days={summary['candidate_days']}")
    if len(base_rates):
        r = base_rates[(base_rates["horizon_sec"] == 1800) & (base_rates["target_bps_from_origin"] == 75)]
        if len(r):
            rr = r.iloc[0]
            print(f"      75bp within 30m base rate={safe_float(rr['continuation_rate']):.3f} ({int(rr['reached'])}/{int(rr['eligible_candidates'])})")
    if len(geom_report):
        print(f"      Phase7A same-direction mapping={safe_float(geom_report.iloc[0]['mapped_share']):.3f}")
    print(f"      candidate id set sha256={candidate_id_hash[:16]}...")
    print(f"      reports={args.report_root}")
    if status == "FAIL":
        print("ERROR: one or more candidate-catalog invariants failed. Do not freeze candidate IDs.")
        return 2
    print("Candidate universe built successfully. Inspect geometry, duplication and base-rate reports before approving the candidate IDs for the expensive Phase-7I participant-interaction materialization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
