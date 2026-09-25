#!/usr/bin/env python3
"""R0.5 — build an outcome-blind D0 causal market-state primitive store.

R0.5 does not search for predictive features.  It freezes reusable causal
primitives and their information clocks so later R1+ work cannot silently
change as-of/window semantics.

Implemented family in v1:
  * Binance BTCUSDT perps canonical trades, receive-time governed.

Deferred rather than guessed:
  * authoritative order-book state (requires governed book replay);
  * OI/funding/basis, spot/cross-venue, and options families (require their own
    source-specific information-availability contracts).

The builder reads R0.3 causal candidates/clocks, R0.4 governance linkage, and
canonical D0 trade partitions only.  It never opens R0.3 future outcomes and
never opens P1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    np = None
    pd = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_core.clocks.contract import (  # noqa: E402
    D0_END_EXCLUSIVE_NS,
    D0_START_NS,
    NS_PER_SECOND,
    REQUIRED_CLOCK_FIELDS,
    assert_clock_contract,
    require_int64_ns,
)
from research_core.market_state import (  # noqa: E402
    TRADE_STORAGE_GRID,
    assert_available_by_cutoff,
    assert_feature_columns_safe,
    latest_prior_index,
    relative_bucket_bounds,
)
from research_core.market_state.schemas import (  # noqa: E402
    BUCKET_FIELDS,
    CANDIDATE_CORE_FIELDS,
    TRADE_STATE_FIELDS,
)

SCHEMA_NAME = "r0_causal_market_state_v1"
SCHEMA_VERSION = "1.0.0"
BUILDER_VERSION = "1.0.1"
R03_SCHEMA = "r0_corrected_d0_substrate_v1"
R04_SCHEMA = "r0_dependence_framework_v1"
SYMBOL = "BTCUSDT"
EXPECTED_CANDIDATES = 7_067
EXPECTED_CLOCKS = 7_031
EXPECTED_CLOCK_UNAVAILABLE = 36
ACCEPTED_R04_COMMIT = "9bc5fd1"
ACCEPTED_R04_TAG = "r0-dependence-framework-v1"

DEFAULT_R03_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / R03_SCHEMA
DEFAULT_R04_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / R04_SCHEMA
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "research" / SCHEMA_NAME
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / SCHEMA_NAME

FEATURE_TABLE_FIELDS = (
    "candidate_id",
    "latest_trade_ts_ns",
    "latest_trade_price",
    "latest_trade_age_ns",
    "latest_trade_tie_ambiguous",
    "trade_source_status",
)
DEPENDENCE_KEEP = (
    "candidate_id",
    "policy_temporal_episode_id",
    "policy_temporal_episode_size",
    "directional_episode_id",
    "directional_episode_size",
    "origin_lineage_episode_id",
    "origin_lineage_episode_size",
    "conservative_inference_cluster_id",
    "conservative_inference_cluster_size",
    "overlap_component_5m",
    "overlap_component_size_5m",
    "overlap_component_10m",
    "overlap_component_size_10m",
    "overlap_component_15m",
    "overlap_component_size_15m",
    "overlap_component_30m",
    "overlap_component_size_30m",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress(msg: str) -> None:
    print(f"[R0.5 {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def require_stack() -> None:
    if pd is None or np is None or pa is None or pq is None:
        raise SystemExit("R0.5 requires numpy, pandas and pyarrow in the project .venv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_parquet(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _require_columns(frame: Any, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def d0_date(day: str) -> bool:
    d = date.fromisoformat(str(day))
    return date(2026, 5, 18) <= d <= date(2026, 7, 18)


def utc_date_from_ns(ts_ns: int) -> str:
    return datetime.fromtimestamp(int(ts_ns) // NS_PER_SECOND, tz=timezone.utc).date().isoformat()


def days_covering_interval(start_ns: int, end_ns: int) -> list[str]:
    """D0 dates intersecting [start,end); raises if interval enters P1."""
    start = require_int64_ns(start_ns, "start_ns")
    end = require_int64_ns(end_ns, "end_ns")
    if end <= start:
        raise ValueError("interval end must be > start")
    if end > D0_END_EXCLUSIVE_NS:
        raise RuntimeError("R0.5 refuses an interval entering P1")
    sday = datetime.fromtimestamp(start // NS_PER_SECOND, tz=timezone.utc).date()
    eday = datetime.fromtimestamp((end - 1) // NS_PER_SECOND, tz=timezone.utc).date()
    out: list[str] = []
    d = sday
    while d <= eday:
        if d >= date(2026, 7, 19):
            raise RuntimeError("R0.5 refuses P1 source date")
        if d >= date(2026, 5, 18):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def canonical_trade_paths(root: Path, day: str) -> list[Path]:
    if not d0_date(day):
        raise RuntimeError(f"R0.5 refuses non-D0 canonical trade date: {day}")
    p = root / "perps" / "trades" / f"date={day}"
    return sorted(p.glob("*.parquet")) if p.exists() else []


def _int64_arrow_field(path: Path, field: str) -> bool:
    t = pq.ParquetFile(path).schema_arrow.field(field).type
    return pa.types.is_int64(t) or pa.types.is_uint64(t)


def load_parents(r03_root: Path, r04_root: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    require_stack()
    r03_manifest_path = r03_root / "metadata" / "manifest.json"
    cpath = r03_root / "causal" / "candidates.parquet"
    kpath = r03_root / "causal" / "clocks.parquet"
    r04_manifest_path = r04_root / "metadata" / "manifest.json"
    dpath = r04_root / "dependence" / "candidate_dependence.parquet"
    for p in (r03_manifest_path, cpath, kpath, r04_manifest_path, dpath):
        if not p.exists():
            raise FileNotFoundError(p)
    r03 = json.loads(r03_manifest_path.read_text(encoding="utf-8"))
    r04 = json.loads(r04_manifest_path.read_text(encoding="utf-8"))
    if r03.get("schema_name") != R03_SCHEMA or r04.get("schema_name") != R04_SCHEMA:
        raise RuntimeError("unexpected R0.3/R0.4 parent schema")
    if bool(r03.get("protected_data", {}).get("p1_read")):
        raise RuntimeError("R0.3 parent reports P1 read")
    if bool(r04.get("outcome_firewall", {}).get("p1_read")):
        raise RuntimeError("R0.4 parent reports P1 read")
    counts = r03.get("counts", {})
    for key, expected in (("candidate_rows", EXPECTED_CANDIDATES), ("clock_rows", EXPECTED_CLOCKS), ("clock_unavailable_rows", EXPECTED_CLOCK_UNAVAILABLE)):
        if int(counts.get(key, -1)) != expected:
            raise RuntimeError(f"R0.5 requires accepted R0.3 {key}={expected}; found {counts.get(key)!r}")
    r04_counts = r04.get("counts", {})
    if int(r04_counts.get("candidate_rows", -1)) != EXPECTED_CANDIDATES:
        raise RuntimeError("R0.4 candidate count mismatch")
    expected_c = str(r03.get("outputs", {}).get("candidates", {}).get("sha256", ""))
    expected_k = str(r03.get("outputs", {}).get("clocks", {}).get("sha256", ""))
    expected_d = str(r04.get("outputs", {}).get("candidate_dependence", {}).get("sha256", ""))
    if sha256_file(cpath) != expected_c or sha256_file(kpath) != expected_k or sha256_file(dpath) != expected_d:
        raise RuntimeError("parent artifact hash mismatch")
    candidates = pq.read_table(cpath).to_pandas()
    clocks = pq.read_table(kpath).to_pandas()
    dep = pq.read_table(dpath).to_pandas()
    _require_columns(candidates, ("candidate_id", "candidate_date", "direction", "clock_available", "clock_unavailable_reason"), "R0.3 candidates")
    _require_columns(clocks, ("candidate_id", *REQUIRED_CLOCK_FIELDS), "R0.3 clocks")
    _require_columns(dep, DEPENDENCE_KEEP, "R0.4 dependence")
    if len(candidates) != EXPECTED_CANDIDATES or len(clocks) != EXPECTED_CLOCKS or len(dep) != EXPECTED_CANDIDATES:
        raise RuntimeError("parent row counts changed")
    if candidates["candidate_id"].astype(str).duplicated().any() or clocks["candidate_id"].astype(str).duplicated().any() or dep["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError("duplicate candidate_id in parent")
    return candidates, clocks, dep, {
        "r03_manifest": str(r03_manifest_path),
        "r03_candidates": {"path": str(cpath), "sha256": expected_c},
        "r03_clocks": {"path": str(kpath), "sha256": expected_k},
        "r04_manifest": str(r04_manifest_path),
        "r04_dependence": {"path": str(dpath), "sha256": expected_d},
    }


def build_candidate_core(candidates: Any, clocks: Any) -> Any:
    clock_by = clocks.set_index(clocks["candidate_id"].astype(str), drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []
    for r in candidates.to_dict("records"):
        cid = str(r["candidate_id"])
        clock = clock_by.get(cid)
        available = bool(r["clock_available"])
        if available != (clock is not None):
            raise RuntimeError(f"clock availability mismatch for {cid}")
        base = {
            "candidate_id": cid,
            "candidate_date": str(r["candidate_date"]),
            "direction": int(r["direction"]),
            "candidate_policy_availability_ts_ns": None,
            "anchor_ts_ns": None,
            "information_available_ts_ns": None,
            "decision_ts_ns": None,
            "clock_available": available,
            "clock_unavailable_reason": None if available else r.get("clock_unavailable_reason"),
        }
        if clock is not None:
            assert_clock_contract(clock)
            base.update({
                "candidate_policy_availability_ts_ns": int(clock["candidate_detection_ts_ns"]),
                "anchor_ts_ns": int(clock["anchor_ts_ns"]),
                "information_available_ts_ns": int(clock["information_available_ts_ns"]),
                "decision_ts_ns": int(clock["decision_ts_ns"]),
            })
        rows.append(base)
    out = pd.DataFrame(rows).sort_values(["candidate_date", "candidate_id"], kind="mergesort").reset_index(drop=True)
    if int(out["clock_available"].astype(bool).sum()) != EXPECTED_CLOCKS:
        raise RuntimeError("candidate core governed-clock count mismatch")
    return out


def merge_half_open_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge half-open intervals for source-scan pruning only.

    Merging touching intervals is safe here because this function controls I/O,
    not scientific bucket membership.
    """
    cleaned = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    out: list[list[int]] = []
    for start, end in cleaned:
        if not out or start > out[-1][1]:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return [(a, b) for a, b in out]


def _row_group_ts_bounds(pf: Any, rg_index: int) -> tuple[int, int] | None:
    """Return exact int64 receive-time bounds from Parquet statistics."""
    names = list(pf.schema_arrow.names)
    if "ts_recv_ns" not in names:
        raise RuntimeError("canonical trade parquet lacks ts_recv_ns")
    col_index = names.index("ts_recv_ns")
    stats = pf.metadata.row_group(rg_index).column(col_index).statistics
    if stats is None or not bool(getattr(stats, "has_min_max", False)):
        return None
    try:
        lo = int(stats.min)
        hi = int(stats.max)
    except Exception:
        return None
    return (lo, hi) if hi >= lo else None


def _interval_overlaps_bounds(intervals: Sequence[tuple[int, int]], bounds: tuple[int, int]) -> bool:
    lo, hi = bounds  # Parquet min/max are inclusive.
    return any(hi >= start and lo < end for start, end in intervals)


def _trade_file_index(path: Path, cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    required = {"ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"canonical trade parquet lacks required fields {missing}: {path}")
    ts_type = pf.schema_arrow.field("ts_recv_ns").type
    if not (pa.types.is_int64(ts_type) or pa.types.is_uint64(ts_type)):
        raise RuntimeError(f"canonical ts_recv_ns is not exact int64: {path}")
    row_groups = []
    ordinal = 0
    for i in range(pf.metadata.num_row_groups):
        n = int(pf.metadata.row_group(i).num_rows)
        row_groups.append({
            "index": i,
            "num_rows": n,
            "ordinal_start": ordinal,
            "bounds": _row_group_ts_bounds(pf, i),
        })
        ordinal += n
    meta = {
        "path": path,
        "columns": sorted(names),
        "num_rows": int(pf.metadata.num_rows),
        "row_groups": row_groups,
    }
    if cache is not None:
        cache[key] = meta
    return meta


def load_trade_intervals(
    canonical_root: Path,
    days: Sequence[str],
    intervals: Sequence[tuple[int, int]],
    *,
    file_index_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]], set[str], dict[str, int]]:
    """Load only Parquet row groups intersecting required causal intervals.

    This is an I/O optimization only.  Scientific selection remains exact in
    ``latest_prior_index`` and ``bucket_from_prefix`` after rows are loaded.
    Row groups without usable timestamp statistics are read conservatively.
    """
    require_stack()
    merged = merge_half_open_intervals(intervals)
    parts = []
    prov: list[dict[str, Any]] = []
    present: set[str] = set()
    stats = {
        "parquet_files_indexed": 0,
        "parquet_files_read": 0,
        "row_groups_total": 0,
        "row_groups_read": 0,
        "rows_total_indexed": 0,
        "rows_read_before_symbol_filter": 0,
    }
    for day in sorted(set(days)):
        paths = canonical_trade_paths(canonical_root, day)
        if not paths:
            continue
        present.add(day)
        for p in paths:
            meta = _trade_file_index(p, file_index_cache)
            stats["parquet_files_indexed"] += 1
            stats["row_groups_total"] += len(meta["row_groups"])
            stats["rows_total_indexed"] += int(meta["num_rows"])
            selected = [
                rg for rg in meta["row_groups"]
                if rg["bounds"] is None or _interval_overlaps_bounds(merged, rg["bounds"])
            ]
            if not selected:
                continue
            stats["parquet_files_read"] += 1
            stats["row_groups_read"] += len(selected)
            tie_cols = [c for c in ("source_file_id", "source_record_number", "aggregate_trade_id") if c in meta["columns"]]
            cols = ["ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker", *tie_cols]
            pf = pq.ParquetFile(p)
            for rg in selected:
                table = pf.read_row_group(int(rg["index"]), columns=cols)
                f = table.to_pandas()
                stats["rows_read_before_symbol_filter"] += len(f)
                # Preserve ordinal in the ORIGINAL Parquet file, not merely in
                # the selected subset, so equal-timestamp ordering is stable.
                f["_source_row_ordinal"] = np.arange(
                    int(rg["ordinal_start"]), int(rg["ordinal_start"]) + len(f), dtype="int64"
                )
                f = f[f["symbol"].astype(str) == SYMBOL].copy()
                if len(f):
                    f["ts_recv_ns"] = pd.to_numeric(f["ts_recv_ns"], errors="raise").astype("int64")
                    # Row-group statistics prune coarse ranges; this exact
                    # filter removes non-required rows at row-group edges.
                    keep = np.zeros(len(f), dtype=bool)
                    vals = f["ts_recv_ns"].to_numpy(dtype="int64", copy=False)
                    for start, end in merged:
                        keep |= (vals >= int(start)) & (vals < int(end))
                    f = f.loc[keep].copy()
                    if len(f):
                        f["price"] = pd.to_numeric(f["price"], errors="coerce")
                        f["quantity"] = pd.to_numeric(f["quantity"], errors="coerce")
                        f = f[f["price"].notna() & (f["price"] > 0) & f["quantity"].notna() & (f["quantity"] > 0)].copy()
                        f["_source_path"] = str(p)
                        parts.append(f)
            prov.append({
                "path": str(p),
                "role": "canonical_perps_trade_pruned",
                "date": day,
                "size_bytes": int(p.stat().st_size),
                "parquet_num_rows": int(meta["num_rows"]),
                "parquet_row_groups_total": len(meta["row_groups"]),
                "parquet_row_groups_read": len(selected),
                "columns": meta["columns"],
            })
    if not parts:
        cols = ["ts_recv_ns", "price", "quantity", "buyer_is_maker", "_source_path", "_source_row_ordinal"]
        return pd.DataFrame(columns=cols), prov, present, stats
    allp = pd.concat(parts, ignore_index=True, sort=False)
    tie_cols = [c for c in ("source_file_id", "source_record_number", "aggregate_trade_id") if c in allp.columns]
    allp = allp.sort_values(["ts_recv_ns", "_source_path", "_source_row_ordinal", *tie_cols], kind="mergesort", na_position="last").reset_index(drop=True)
    return allp, prov, present, stats


def load_trade_days(canonical_root: Path, days: Sequence[str]) -> tuple[Any, list[dict[str, Any]], set[str]]:
    """Load canonical trade rows for D0 days, preserving exact receive timestamps."""
    require_stack()
    parts = []
    prov: list[dict[str, Any]] = []
    present: set[str] = set()
    for day in sorted(set(days)):
        paths = canonical_trade_paths(canonical_root, day)
        if not paths:
            continue
        present.add(day)
        for p in paths:
            pf = pq.ParquetFile(p)
            names = set(pf.schema_arrow.names)
            required = {"ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker"}
            missing = sorted(required - names)
            if missing:
                raise RuntimeError(f"canonical trade parquet lacks required fields {missing}: {p}")
            if not _int64_arrow_field(p, "ts_recv_ns"):
                raise RuntimeError(f"canonical ts_recv_ns is not exact int64: {p}")
            tie_cols = [c for c in ("source_file_id", "source_record_number", "aggregate_trade_id") if c in names]
            cols = ["ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker", *tie_cols]
            f = pq.read_table(p, columns=cols).to_pandas()
            # Parquet row order is part of the consumed source representation.
            # Persist an explicit ordinal before filtering so equal receive-time
            # trades are deterministically ordered even when optional exchange
            # tie fields are absent.
            f["_source_row_ordinal"] = np.arange(len(f), dtype="int64")
            f = f[f["symbol"].astype(str) == SYMBOL].copy()
            if len(f):
                f["ts_recv_ns"] = pd.to_numeric(f["ts_recv_ns"], errors="raise").astype("int64")
                f["price"] = pd.to_numeric(f["price"], errors="coerce")
                f["quantity"] = pd.to_numeric(f["quantity"], errors="coerce")
                f = f[f["price"].notna() & (f["price"] > 0) & f["quantity"].notna() & (f["quantity"] > 0)].copy()
                f["_source_path"] = str(p)
                parts.append(f)
            prov.append({
                "path": str(p),
                "role": "canonical_perps_trade",
                "date": day,
                "size_bytes": int(p.stat().st_size),
                "parquet_num_rows": int(pf.metadata.num_rows),
                "columns": sorted(names),
            })
    if not parts:
        cols = ["ts_recv_ns", "price", "quantity", "buyer_is_maker", "_source_path"]
        return pd.DataFrame(columns=cols), prov, present
    allp = pd.concat(parts, ignore_index=True, sort=False)
    tie_cols = [c for c in ("source_file_id", "source_record_number", "aggregate_trade_id") if c in allp.columns]
    allp = allp.sort_values(["ts_recv_ns", "_source_path", "_source_row_ordinal", *tie_cols], kind="mergesort", na_position="last").reset_index(drop=True)
    return allp, prov, present


def bool_mask(values: Any, wanted: bool) -> Any:
    text = values.astype(str).str.strip().str.lower()
    if wanted:
        return text.isin(["true", "1", "t"])
    return text.isin(["false", "0", "f"])


def prepare_trade_arrays(frame: Any) -> dict[str, Any]:
    if frame.empty:
        return {
            "ts": [], "price": [], "qty": [], "maker": [], "tie_keys": None,
            "prefix_qty": np.asarray([0.0]), "prefix_notional": np.asarray([0.0]),
            "prefix_true": np.asarray([0.0]), "prefix_false": np.asarray([0.0]),
            "prefix_unknown": np.asarray([0.0]),
        }
    ts = frame["ts_recv_ns"].astype("int64").to_numpy(copy=False)
    price = frame["price"].astype(float).to_numpy(copy=False)
    qty = frame["quantity"].astype(float).to_numpy(copy=False)
    maker_true = bool_mask(frame["buyer_is_maker"], True).to_numpy(dtype=bool)
    maker_false = bool_mask(frame["buyer_is_maker"], False).to_numpy(dtype=bool)
    maker_unknown = ~(maker_true | maker_false)
    # Source path + immutable row ordinal provide a deterministic tie break for
    # equal ts_recv_ns rows without collapsing legitimate trades. Optional
    # exchange/source IDs are appended when present.
    tie_cols = ["_source_path", "_source_row_ordinal", *[c for c in ("source_file_id", "source_record_number", "aggregate_trade_id") if c in frame.columns]]
    tie_keys = list(zip(*(frame[c].astype(str).tolist() for c in tie_cols)))
    def pref(x: Any) -> Any:
        return np.concatenate([np.asarray([0.0]), np.cumsum(np.asarray(x, dtype=float))])
    return {
        "ts": ts.tolist(),
        "price": price.tolist(),
        "qty": qty.tolist(),
        "maker": frame["buyer_is_maker"].tolist(),
        "tie_keys": tie_keys,
        "prefix_qty": pref(qty),
        "prefix_notional": pref(price * qty),
        "prefix_true": pref(qty * maker_true),
        "prefix_false": pref(qty * maker_false),
        "prefix_unknown": pref(qty * maker_unknown),
    }


def interval_partition_status(start_ns: int, end_ns: int, present_days: set[str]) -> str:
    if start_ns < D0_START_NS:
        return "OUTSIDE_D0_HISTORY"
    days = days_covering_interval(start_ns, end_ns)
    missing = [d for d in days if d not in present_days]
    return "SOURCE_PARTITION_MISSING" if missing else "AVAILABLE"


def bucket_from_prefix(arr: Mapping[str, Any], start_ns: int, end_ns: int) -> dict[str, Any]:
    from bisect import bisect_left
    ts = arr["ts"]
    lo = bisect_left(ts, int(start_ns))
    hi = bisect_left(ts, int(end_ns))
    def delta(name: str) -> float:
        p = arr[name]
        return float(p[hi] - p[lo])
    return {
        "trade_count": int(hi - lo),
        "quantity_sum": delta("prefix_qty"),
        "notional_sum": delta("prefix_notional"),
        "buyer_maker_true_quantity": delta("prefix_true"),
        "buyer_maker_false_quantity": delta("prefix_false"),
        "buyer_maker_unknown_quantity": delta("prefix_unknown"),
        "first_trade_ts_ns": None if hi <= lo else int(ts[lo]),
        "last_trade_ts_ns": None if hi <= lo else int(ts[hi - 1]),
    }


def build_trade_state_and_buckets(candidate_core: Any, canonical_root: Path, progress_every_day: int = 5) -> tuple[Any, Any, Any, Any, list[dict[str, Any]], dict[str, int]]:
    """Build latest-prior trade state plus candidate-relative 1s/1m additive buckets.

    Performance rule: canonical Parquet row groups are pruned to the union of
    each day's required 30-minute causal histories.  If a candidate has no
    trade in that pruned history, the function falls back to the original full
    current+previous-day loader for that day, preserving exact latest-prior
    semantics rather than turning the optimization into a scientific change.
    """
    governed = candidate_core[candidate_core["clock_available"].astype(bool)].copy()
    governed["cutoff_day"] = governed["information_available_ts_ns"].map(lambda x: utc_date_from_ns(int(x)))
    core_by_day = {d: g.copy() for d, g in governed.groupby("cutoff_day", sort=True)}
    state_rows: list[dict[str, Any]] = []
    fine_rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    scan_totals: Counter[str] = Counter()
    file_index_cache: dict[str, dict[str, Any]] = {}

    unavailable = candidate_core[~candidate_core["clock_available"].astype(bool)]
    for r in unavailable.to_dict("records"):
        state_rows.append({
            "candidate_id": str(r["candidate_id"]),
            "latest_trade_ts_ns": None,
            "latest_trade_price": None,
            "latest_trade_age_ns": None,
            "latest_trade_tie_ambiguous": False,
            "trade_source_status": "CLOCK_UNAVAILABLE",
        })
        coverage_rows.append({"candidate_id": str(r["candidate_id"]), "clock_available": False, "fine_history_status": "CLOCK_UNAVAILABLE", "coarse_history_status": "CLOCK_UNAVAILABLE"})

    coarse_lookback_ns = int(TRADE_STORAGE_GRID["coarse"]["lookback_sec"]) * NS_PER_SECOND
    days = sorted(core_by_day)
    for day_i, (day, group) in enumerate(sorted(core_by_day.items()), start=1):
        day_date = date.fromisoformat(day)
        prev = (day_date - timedelta(days=1)).isoformat()
        load_days = [day]
        if d0_date(prev):
            load_days.insert(0, prev)

        # One merged I/O plan per candidate day.  +1ns admits a trade stamped
        # exactly at cutoff for latest-prior state; bucket code still excludes
        # that row through its half-open [start, cutoff) boundaries.
        raw_intervals = []
        for cutoff_raw in group["information_available_ts_ns"].tolist():
            cutoff = require_int64_ns(int(cutoff_raw), "information_available_ts_ns")
            raw_intervals.append((max(D0_START_NS, cutoff - coarse_lookback_ns), cutoff + 1))
        merged_intervals = merge_half_open_intervals(raw_intervals)
        trades, prov, present, day_scan = load_trade_intervals(
            canonical_root, load_days, merged_intervals, file_index_cache=file_index_cache
        )
        provenance.extend(prov)
        scan_totals.update(day_scan)
        arr = prepare_trade_arrays(trades)
        ts = arr["ts"]
        prices = arr["price"]

        ordered_group = group.sort_values(["information_available_ts_ns", "candidate_id"], kind="mergesort").to_dict("records")
        # Exactness fallback: if any candidate has no BTC trade inside its
        # selected 30m history, use the original full-day loader for the whole
        # day group.  This is expected to be exceptional for BTCUSDT but makes
        # the optimization semantics-preserving by construction.
        needs_full_fallback = False
        from bisect import bisect_left, bisect_right
        for r in ordered_group:
            cutoff = int(r["information_available_ts_ns"])
            history_start = max(D0_START_NS, cutoff - coarse_lookback_ns)
            if bisect_right(ts, cutoff) <= bisect_left(ts, history_start):
                needs_full_fallback = True
                break
        if needs_full_fallback:
            progress(f"trade state: {day} requires exact latest-prior fallback to full current/previous D0 day")
            trades, prov_full, present = load_trade_days(canonical_root, load_days)
            provenance.extend(prov_full)
            arr = prepare_trade_arrays(trades)
            ts = arr["ts"]
            prices = arr["price"]
            scan_totals["full_day_fallback_groups"] += 1

        if day_i == 1 or day_i % progress_every_day == 0 or day_i == len(days):
            progress(
                f"trade state: D0 day {day_i}/{len(days)} ({day}); "
                f"rowgroups read {day_scan['row_groups_read']}/{day_scan['row_groups_total']}, "
                f"files read {day_scan['parquet_files_read']}/{day_scan['parquet_files_indexed']}"
            )

        for r in ordered_group:
            cid = str(r["candidate_id"])
            cutoff = require_int64_ns(int(r["information_available_ts_ns"]), "information_available_ts_ns")
            sel = latest_prior_index(ts, cutoff, deterministic_tie_keys=arr["tie_keys"])
            cutoff_day = utc_date_from_ns(cutoff)
            if cutoff_day not in present:
                state_rows.append({"candidate_id": cid, "latest_trade_ts_ns": None, "latest_trade_price": None, "latest_trade_age_ns": None, "latest_trade_tie_ambiguous": False, "trade_source_status": "SOURCE_PARTITION_MISSING"})
            elif sel is None:
                state_status = "NO_PRIOR_TRADE_IN_LOADED_D0_WINDOW"
                state_rows.append({"candidate_id": cid, "latest_trade_ts_ns": None, "latest_trade_price": None, "latest_trade_age_ns": None, "latest_trade_tie_ambiguous": False, "trade_source_status": state_status})
            else:
                assert_available_by_cutoff(sel.ts_ns, cutoff)
                interval_days = days_covering_interval(max(D0_START_NS, int(sel.ts_ns)), cutoff + 1)
                missing_between = [d for d in interval_days if d not in present]
                if missing_between:
                    state_rows.append({"candidate_id": cid, "latest_trade_ts_ns": None, "latest_trade_price": None, "latest_trade_age_ns": None, "latest_trade_tie_ambiguous": False, "trade_source_status": "SOURCE_PARTITION_MISSING"})
                else:
                    state_rows.append({
                        "candidate_id": cid,
                        "latest_trade_ts_ns": int(sel.ts_ns),
                        "latest_trade_price": None if sel.tie_ambiguous else float(prices[sel.index]),
                        "latest_trade_age_ns": int(sel.age_ns),
                        "latest_trade_tie_ambiguous": bool(sel.tie_ambiguous),
                        "trade_source_status": "AVAILABLE" if not sel.tie_ambiguous else "LATEST_TIE_AMBIGUOUS",
                    })

            statuses: dict[str, str] = {}
            for grid_name, spec, sink in (
                ("fine", TRADE_STORAGE_GRID["fine"], fine_rows),
                ("coarse", TRADE_STORAGE_GRID["coarse"], coarse_rows),
            ):
                bounds = relative_bucket_bounds(cutoff, lookback_sec=int(spec["lookback_sec"]), bucket_sec=int(spec["bucket_sec"]))
                history_status = interval_partition_status(bounds[0][1], cutoff, present)
                statuses[grid_name] = history_status
                for offset_sec, start_ns, end_ns in bounds:
                    row = {
                        "candidate_id": cid,
                        "cutoff_ts_ns": cutoff,
                        "bucket_offset_sec": int(offset_sec),
                        "bucket_start_ts_ns": int(start_ns),
                        "bucket_end_ts_ns": int(end_ns),
                        "bucket_width_sec": int(spec["bucket_sec"]),
                        "source_status": history_status,
                    }
                    if history_status == "AVAILABLE":
                        row.update(bucket_from_prefix(arr, start_ns, end_ns))
                    else:
                        row.update({
                            "trade_count": None,
                            "quantity_sum": None,
                            "notional_sum": None,
                            "buyer_maker_true_quantity": None,
                            "buyer_maker_false_quantity": None,
                            "buyer_maker_unknown_quantity": None,
                            "first_trade_ts_ns": None,
                            "last_trade_ts_ns": None,
                        })
                    sink.append(row)
            coverage_rows.append({"candidate_id": cid, "clock_available": True, "fine_history_status": statuses["fine"], "coarse_history_status": statuses["coarse"]})

    state = pd.DataFrame(state_rows).sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    fine = pd.DataFrame(fine_rows).sort_values(["candidate_id", "bucket_start_ts_ns"], kind="mergesort").reset_index(drop=True)
    coarse = pd.DataFrame(coarse_rows).sort_values(["candidate_id", "bucket_start_ts_ns"], kind="mergesort").reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows).sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    return state, fine, coarse, coverage, provenance, {k: int(v) for k, v in scan_totals.items()}


def build_governance_linkage(dep: Any) -> Any:
    out = dep[list(DEPENDENCE_KEEP)].copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    return out.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def field_registry() -> Any:
    rows = []
    for table, specs in (("candidate_core", CANDIDATE_CORE_FIELDS), ("trade_state", TRADE_STATE_FIELDS), ("trade_buckets", BUCKET_FIELDS)):
        for spec in specs:
            rows.append({"table": table, "field": spec.name, "role": spec.role, "information_clock": spec.information_clock, "description": spec.description})
    structural = {
        "trade_buckets": {
            "candidate_id": "IDENTITY_ONLY", "cutoff_ts_ns": "QUALITY_ONLY",
            "bucket_offset_sec": "QUALITY_ONLY", "bucket_start_ts_ns": "QUALITY_ONLY",
            "bucket_end_ts_ns": "QUALITY_ONLY", "bucket_width_sec": "QUALITY_ONLY",
            "source_status": "QUALITY_ONLY",
        },
        "trade_state": {"candidate_id": "IDENTITY_ONLY"},
    }
    existing = {(r["table"], r["field"]) for r in rows}
    for table, fields in structural.items():
        for field, role in fields.items():
            if (table, field) not in existing:
                rows.append({"table": table, "field": field, "role": role, "information_clock": "R0.5 structural", "description": "Storage/provenance field; not an automatically authorized scientific predictor."})
    for field in DEPENDENCE_KEEP:
        rows.append({"table": "governance_dependence", "field": field, "role": "GOVERNANCE_ONLY" if field != "candidate_id" else "IDENTITY_ONLY", "information_clock": "R0.4", "description": "Inference/splitting metadata; prohibited as a predictive feature."})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    require_stack()
    progress("stage 1/5: validating frozen R0.3 causal parents and accepted R0.4 dependence artifact")
    candidates, clocks, dep, parent_prov = load_parents(args.r03_root, args.r04_root)
    core = build_candidate_core(candidates, clocks)
    governance = build_governance_linkage(dep)
    if set(core["candidate_id"].astype(str)) != set(governance["candidate_id"].astype(str)):
        raise RuntimeError("R0.3/R0.4 candidate identity sets differ")
    progress(f"stage 1/5 complete: {len(core):,} candidates, {int(core['clock_available'].astype(bool).sum()):,} governed clocks")

    progress("stage 2/5: building receive-time causal perps trade state and multiresolution storage buckets")
    trade_state, fine, coarse, coverage, source_prov, scan_stats = build_trade_state_and_buckets(core, args.canonical_root, args.progress_every_day)
    progress(f"stage 2/5 complete: {len(trade_state):,} trade-state rows, {len(fine):,} fine buckets, {len(coarse):,} coarse buckets")

    progress("stage 3/5: validating causal cutoffs, physical separation, and preservation invariants")
    if len(core) != EXPECTED_CANDIDATES or len(trade_state) != EXPECTED_CANDIDATES or len(governance) != EXPECTED_CANDIDATES:
        raise AssertionError("candidate preservation failed")
    if len(fine) != EXPECTED_CLOCKS * (TRADE_STORAGE_GRID["fine"]["lookback_sec"] // TRADE_STORAGE_GRID["fine"]["bucket_sec"]):
        raise AssertionError("fine bucket row count mismatch")
    if len(coarse) != EXPECTED_CLOCKS * (TRADE_STORAGE_GRID["coarse"]["lookback_sec"] // TRADE_STORAGE_GRID["coarse"]["bucket_sec"]):
        raise AssertionError("coarse bucket row count mismatch")
    # Only values in actual causal feature tables are checked here; governance
    # columns are deliberately stored elsewhere.
    assert_feature_columns_safe(FEATURE_TABLE_FIELDS)
    if len(set(governance.columns) & set(trade_state.columns)) > 1:  # candidate_id is the only allowed overlap
        raise AssertionError("governance fields leaked into causal trade state")
    available_state = trade_state[trade_state["latest_trade_ts_ns"].notna()]
    core_cutoff = core.set_index("candidate_id")["information_available_ts_ns"].to_dict()
    for r in available_state.to_dict("records"):
        assert_available_by_cutoff(int(r["latest_trade_ts_ns"]), int(core_cutoff[str(r["candidate_id"]) ]))
    if len(fine) and bool((fine["bucket_end_ts_ns"] > fine["cutoff_ts_ns"]).any()):
        raise AssertionError("fine bucket extends after causal cutoff")
    if len(coarse) and bool((coarse["bucket_end_ts_ns"] > coarse["cutoff_ts_ns"]).any()):
        raise AssertionError("coarse bucket extends after causal cutoff")
    progress("stage 3/5 complete: causal-state invariants pass; future outcomes remain unopened")

    progress("stage 4/5: writing causal, governance, quality and field-registry artifacts atomically")
    paths = {
        "candidate_core": args.output_root / "causal" / "candidate_core.parquet",
        "trade_state": args.output_root / "causal" / "trade_state.parquet",
        "trade_buckets_1s": args.output_root / "causal" / "trade_buckets_1s.parquet",
        "trade_buckets_1m": args.output_root / "causal" / "trade_buckets_1m.parquet",
        "dependence_linkage": args.output_root / "governance" / "dependence_linkage.parquet",
        "coverage": args.output_root / "quality" / "trade_coverage.parquet",
    }
    atomic_parquet(paths["candidate_core"], core)
    atomic_parquet(paths["trade_state"], trade_state)
    atomic_parquet(paths["trade_buckets_1s"], fine)
    atomic_parquet(paths["trade_buckets_1m"], coarse)
    atomic_parquet(paths["dependence_linkage"], governance)
    atomic_parquet(paths["coverage"], coverage)
    registry = field_registry()
    atomic_csv(args.report_root / "field_registry.csv", registry)
    source_df = pd.DataFrame(source_prov).drop_duplicates(subset=["path"], keep="first") if source_prov else pd.DataFrame(columns=["path", "role", "date", "size_bytes", "parquet_num_rows", "columns"])
    if len(source_df):
        source_df = source_df.copy()
        source_df["columns"] = source_df["columns"].map(lambda x: json.dumps(x, separators=(",", ":")))
    atomic_csv(args.report_root / "source_provenance.csv", source_df)
    status_counts = {str(k): int(v) for k, v in trade_state["trade_source_status"].fillna("UNKNOWN").value_counts().items()}
    fine_cov = {str(k): int(v) for k, v in coverage["fine_history_status"].fillna("UNKNOWN").value_counts().items()}
    coarse_cov = {str(k): int(v) for k, v in coverage["coarse_history_status"].fillna("UNKNOWN").value_counts().items()}
    progress("stage 4/5 complete")

    progress("stage 5/5: writing manifest and outcome-blind diagnostics")
    output_meta = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "generated_at_utc": utc_now_iso(),
        "evidentiary_role": "D0_CAUSAL_MARKET_STATE_INFRASTRUCTURE_ONLY",
        "accepted_parent_checkpoint": {"commit": ACCEPTED_R04_COMMIT, "tag": ACCEPTED_R04_TAG},
        "parents": parent_prov,
        "canonical": {
            "logical_root": str(args.canonical_root),
            "r0_1_lineage_tag": "r0-governance-provenance-v1",
            "family_read": "perps/trades only",
            "source_files_observed": int(len(source_df)),
            "source_full_content_rehash": False,
            "performance_scan": scan_stats,
            "reason": "R0.1 already froze canonical provenance; R0.5 records exact consumed paths/schema/size and hashes its own outputs.",
        },
        "outcome_firewall": {
            "r0_3_future_outcomes_opened": False,
            "returns_mfe_mae_threshold_labels_used": False,
            "feature_selection_from_outcomes": False,
            "p1_read": False,
        },
        "causal_contract": {
            "feature_cutoff": "R0.3 information_available_ts_ns",
            "asof_rule": "latest canonical receive-time trade with ts_recv_ns <= cutoff; no future nearest-neighbor/interpolation",
            "window_rule": "candidate-relative half-open [start, cutoff); a trade exactly at cutoff is excluded from buckets but eligible for latest-prior state",
            "nanosecond_rule": "int64 only",
            "trade_storage_grid": TRADE_STORAGE_GRID,
            "storage_grid_is_scientific_feature_definition": False,
            "future_hypothesis_rule": "R1+ must register exact windows/transforms; R0.5 storage resolutions do not authorize a scientific threshold.",
        },
        "family_registry": {
            "PERPS_TRADES": {"status": "IMPLEMENTED", "clock": "canonical ts_recv_ns"},
            "ORDER_BOOK": {"status": "DEFERRED", "reason": "requires authoritative snapshot+diff replay and explicit book-validity/freshness contract; legacy derived features are not silently imported"},
            "PERPS_DERIVATIVES": {"status": "DEFERRED", "reason": "OI/funding/mark/index require source-specific publication/receive clocks"},
            "SPOT_CROSS_VENUE": {"status": "DEFERRED", "reason": "requires venue-specific causal clocks"},
            "OPTIONS": {"status": "DEFERRED", "reason": "requires option surface/OI/Greek freshness and availability contracts"},
        },
        "counts": {
            "candidate_rows": int(len(core)),
            "governed_clock_rows": int(core["clock_available"].astype(bool).sum()),
            "clock_unavailable_rows": int((~core["clock_available"].astype(bool)).sum()),
            "trade_state_rows": int(len(trade_state)),
            "fine_bucket_rows": int(len(fine)),
            "coarse_bucket_rows": int(len(coarse)),
            "trade_state_status_counts": status_counts,
            "fine_history_status_counts": fine_cov,
            "coarse_history_status_counts": coarse_cov,
            "latest_trade_tie_ambiguous_rows": int(trade_state["latest_trade_tie_ambiguous"].fillna(False).astype(bool).sum()),
        },
        "outputs": output_meta,
        "reports": {
            "field_registry": str(args.report_root / "field_registry.csv"),
            "source_provenance": str(args.report_root / "source_provenance.csv"),
        },
        "status": "BUILD_COMPLETE_REVIEW_REQUIRED",
    }
    manifest_path = args.output_root / "metadata" / "manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(args.report_root / "summary.json", manifest)
    progress("stage 5/5 complete")
    print(json.dumps({"status": manifest["status"], **manifest["counts"], "implemented_families": ["PERPS_TRADES"], "deferred_families": ["ORDER_BOOK", "PERPS_DERIVATIVES", "SPOT_CROSS_VENUE", "OPTIONS"]}, indent=2, sort_keys=True), flush=True)
    return 0


def self_test() -> int:
    from research_core.market_state import latest_prior_index, relative_bucket_bounds, summarize_trade_bucket
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    ts = [cutoff - 2 * NS_PER_SECOND, cutoff - 1, cutoff, cutoff + 1]
    sel = latest_prior_index(ts, cutoff, deterministic_tie_keys=[0, 1, 2, 3])
    assert sel is not None and sel.ts_ns == cutoff
    bounds = relative_bucket_bounds(cutoff, lookback_sec=2, bucket_sec=1)
    assert bounds[-1][2] == cutoff
    b = summarize_trade_bucket(
        trade_ts_ns=ts,
        trade_prices=[100.0, 101.0, 102.0, 103.0],
        trade_quantities=[1.0, 2.0, 3.0, 4.0],
        buyer_is_maker=[True, False, True, False],
        start_ts_ns=cutoff - NS_PER_SECOND,
        end_ts_ns=cutoff,
    )
    assert b.trade_count == 1 and b.quantity_sum == 2.0  # exact-cutoff row excluded
    try:
        latest_prior_index([float(cutoff)], cutoff)
    except Exception as exc:
        if "CLOCK_NOT_INT64_NS" not in str(exc):
            raise
    else:
        raise AssertionError("float epoch ns must be rejected")
    print("SELF-TEST PASS: exact int64 as-of cutoff, half-open causal buckets, exact-cutoff separation, and no-future selection validated.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r03-root", type=Path, default=DEFAULT_R03_ROOT)
    p.add_argument("--r04-root", type=Path, default=DEFAULT_R04_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--progress-every-day", type=int, default=5)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.progress_every_day <= 0:
        raise SystemExit("--progress-every-day must be > 0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
