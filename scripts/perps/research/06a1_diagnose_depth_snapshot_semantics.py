#!/usr/bin/env python3
"""Diagnose Phase-6A BTCUSDT stored depth-snapshot semantics.

This is a read-only diagnostic. It does not modify canonical data and it does
not change the Phase-6A reconstruction contract. Its purpose is to explain why
most stored snapshots on the Phase-6A smoke day were rejected as invalid.

For every snapshot it records:
  * raw/usable bid and ask level counts,
  * precise top status (VALID / EMPTY_* / LOCKED / CROSSED / malformed),
  * normal-side vs swapped-side interpretation,
  * stored-order diagnostics (bid descending / ask ascending),
  * first-listed top vs extrema-derived top,
  * first depth update before/after the snapshot and bridge compatibility,
  * comparison with a shadow book reconstructed only from a known-good
    snapshot and a continuous pu->u diff chain.

The shadow book deliberately ignores subsequent stored snapshots while its diff
chain remains continuous. Therefore it can answer the key diagnostic question:
"Is a suspect snapshot inconsistent with an otherwise continuous diff book?"
without silently changing the production reconstruction algorithm.

Default smoke day: 2026-06-12.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import json
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

DIAGNOSTIC_VERSION = "1.0.0"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_SMOKE_DATE = date(2026, 6, 12)


def find_repo_root(script_path: Path) -> Path:
    p = script_path.resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "scripts").exists():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "snapshot_semantics_diagnostic_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_snapshot_semantics_diagnostic"


def require_data_stack():
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit("06A1 requires numpy, pandas and pyarrow. Activate the project .venv first.") from exc
    return np, pd, pa, pq


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name[:12]}-{uuid.uuid4().hex[:8]}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    tmp = path.parent / f".{path.name[:12]}-{uuid.uuid4().hex[:8]}.tmp"
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def discover_family_files(root: Path, family: str, d: date) -> tuple[Path, ...]:
    base = root / "perps" / family / f"date={d.isoformat()}"
    if not base.exists():
        return ()
    return tuple(sorted(p for p in base.rglob("*.parquet") if p.is_file()))


def level_pair(level: Any) -> tuple[float | None, float | None]:
    if isinstance(level, Mapping):
        p = safe_float(level.get("price"))
        q = safe_float(level.get("quantity"))
        if q is None:
            q = safe_float(level.get("qty"))
        return p, q
    try:
        return safe_float(level[0]), safe_float(level[1])
    except Exception:
        return None, None


def parse_levels(levels: Any) -> dict[str, Any]:
    raw = list(levels or [])
    usable: list[tuple[float, float]] = []
    malformed = 0
    nonpositive = 0
    for level in raw:
        p, q = level_pair(level)
        if p is None or q is None:
            malformed += 1
            continue
        if p <= 0 or q <= 0:
            nonpositive += 1
            continue
        usable.append((p, q))
    prices = [p for p, _ in usable]
    first_price = usable[0][0] if usable else None
    return {
        "raw_count": len(raw),
        "usable_count": len(usable),
        "malformed_count": malformed,
        "nonpositive_count": nonpositive,
        "pairs": usable,
        "prices": prices,
        "first_price": first_price,
    }


def monotonic_fraction(prices: Sequence[float], descending: bool) -> float:
    if len(prices) <= 1:
        return 1.0 if prices else math.nan
    good = 0
    total = len(prices) - 1
    for a, b in zip(prices[:-1], prices[1:]):
        if (a >= b) if descending else (a <= b):
            good += 1
    return good / total


def classify_top(bid: float | None, ask: float | None) -> str:
    if bid is None and ask is None:
        return "EMPTY_BOTH"
    if bid is None:
        return "EMPTY_BID"
    if ask is None:
        return "EMPTY_ASK"
    if bid <= 0 or ask <= 0:
        return "NON_POSITIVE_TOP"
    if bid < ask:
        return "VALID"
    if bid == ask:
        return "LOCKED"
    return "CROSSED"


def cross_bps(bid: float | None, ask: float | None) -> float:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return math.nan
    mid = (bid + ask) / 2.0
    return (bid - ask) / mid * 10_000.0 if mid > 0 else math.nan


def midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def snapshot_semantics(row: Mapping[str, Any]) -> dict[str, Any]:
    bids = parse_levels(row.get("bids"))
    asks = parse_levels(row.get("asks"))
    bp = bids["prices"]
    ap = asks["prices"]

    normal_bid = max(bp) if bp else None
    normal_ask = min(ap) if ap else None
    normal_status = classify_top(normal_bid, normal_ask)

    # Hypothesis: canonical side labels are reversed.
    swapped_bid = max(ap) if ap else None
    swapped_ask = min(bp) if bp else None
    swapped_status = classify_top(swapped_bid, swapped_ask)

    # Hypothesis: arrays carry a meaningful top as their first element even if
    # deeper levels are not a coherent full book.
    first_bid = bids["first_price"]
    first_ask = asks["first_price"]
    first_status = classify_top(first_bid, first_ask)

    bid_desc = monotonic_fraction(bp, descending=True)
    ask_asc = monotonic_fraction(ap, descending=False)
    bid_asc = monotonic_fraction(bp, descending=False)
    ask_desc = monotonic_fraction(ap, descending=True)

    # Count levels that would sit on the wrong side of the extrema-derived top.
    crossed_bid_levels = 0
    crossed_ask_levels = 0
    if normal_ask is not None:
        crossed_bid_levels = sum(1 for p in bp if p >= normal_ask)
    if normal_bid is not None:
        crossed_ask_levels = sum(1 for p in ap if p <= normal_bid)

    return {
        "bid_raw_count": bids["raw_count"],
        "ask_raw_count": asks["raw_count"],
        "bid_usable_count": bids["usable_count"],
        "ask_usable_count": asks["usable_count"],
        "bid_malformed_count": bids["malformed_count"],
        "ask_malformed_count": asks["malformed_count"],
        "bid_nonpositive_count": bids["nonpositive_count"],
        "ask_nonpositive_count": asks["nonpositive_count"],
        "normal_best_bid": normal_bid,
        "normal_best_ask": normal_ask,
        "normal_status": normal_status,
        "normal_cross_bps": cross_bps(normal_bid, normal_ask),
        "swapped_best_bid": swapped_bid,
        "swapped_best_ask": swapped_ask,
        "swapped_status": swapped_status,
        "swapped_cross_bps": cross_bps(swapped_bid, swapped_ask),
        "first_listed_bid": first_bid,
        "first_listed_ask": first_ask,
        "first_listed_status": first_status,
        "first_listed_cross_bps": cross_bps(first_bid, first_ask),
        "bid_descending_fraction": bid_desc,
        "ask_ascending_fraction": ask_asc,
        "bid_ascending_fraction": bid_asc,
        "ask_descending_fraction": ask_desc,
        "crossed_bid_level_count": crossed_bid_levels,
        "crossed_ask_level_count": crossed_ask_levels,
        "normal_mid": midpoint(normal_bid, normal_ask),
        "swapped_mid": midpoint(swapped_bid, swapped_ask),
    }


def bridge_classification(last_u: int | None, depth: Mapping[str, Any] | None) -> str:
    if last_u is None:
        return "NO_SNAPSHOT_ID"
    if depth is None:
        return "NO_NEXT_DIFF"
    U = safe_int(depth.get("first_update_id"))
    u = safe_int(depth.get("final_update_id"))
    pu = safe_int(depth.get("previous_final_update_id"))
    if U is None or u is None or pu is None:
        return "NEXT_DIFF_IDS_MISSING"
    if u < last_u:
        return "NEXT_DIFF_STALE"
    if pu == last_u:
        return "LOCAL_CHECKPOINT_COMPATIBLE"
    if U <= last_u <= u:
        return "REST_BRIDGE_COMPATIBLE"
    return "NO_SEQUENCE_BRIDGE"


class BookSide:
    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.prices: list[float] = []
        self.qty: dict[float, float] = {}

    def reset(self, levels: Sequence[Any]) -> None:
        self.prices.clear()
        self.qty.clear()
        for level in levels or []:
            p, q = level_pair(level)
            if p is None or q is None or p <= 0 or q <= 0:
                continue
            if p not in self.qty:
                bisect.insort(self.prices, p)
            self.qty[p] = q

    def update(self, price: float, qty: float) -> None:
        if not math.isfinite(price) or price <= 0 or not math.isfinite(qty) or qty < 0:
            return
        exists = price in self.qty
        if qty == 0:
            if exists:
                del self.qty[price]
                i = bisect.bisect_left(self.prices, price)
                if i < len(self.prices) and self.prices[i] == price:
                    self.prices.pop(i)
            return
        if not exists:
            bisect.insort(self.prices, price)
        self.qty[price] = qty

    def best(self) -> float | None:
        if not self.prices:
            return None
        return self.prices[-1] if self.is_bid else self.prices[0]


class ShadowBook:
    """Diagnostic diff-chain book seeded only from a normal-valid snapshot."""

    def __init__(self):
        self.bids = BookSide(True)
        self.asks = BookSide(False)
        self.valid = False
        self.bridge_pending = False
        self.seed_last_u: int | None = None
        self.previous_u: int | None = None
        self.last_state_ts_ns: int | None = None
        self.seed_ts_ns: int | None = None
        self.sequence_gap_count = 0
        self.crossed_after_diff_count = 0

    def top(self) -> tuple[float | None, float | None]:
        return self.bids.best(), self.asks.best()

    def top_valid(self) -> bool:
        b, a = self.top()
        return classify_top(b, a) == "VALID"

    def seed(self, row: Mapping[str, Any], sem: Mapping[str, Any]) -> None:
        last_u = safe_int(row.get("last_update_id"))
        if sem.get("normal_status") != "VALID" or last_u is None:
            return
        self.bids.reset(row.get("bids") or [])
        self.asks.reset(row.get("asks") or [])
        self.valid = self.top_valid()
        self.bridge_pending = self.valid
        self.seed_last_u = last_u
        self.previous_u = None
        ts = safe_int(row.get("ts_recv_ns"))
        self.last_state_ts_ns = ts
        self.seed_ts_ns = ts

    def apply_levels(self, bids: Sequence[Any], asks: Sequence[Any]) -> None:
        for side, levels in ((self.bids, bids or []), (self.asks, asks or [])):
            for level in levels:
                p, q = level_pair(level)
                if p is not None and q is not None:
                    side.update(p, q)

    def apply_depth(self, row: Mapping[str, Any]) -> str:
        if not self.valid:
            return "SHADOW_INVALID"
        U = safe_int(row.get("first_update_id"))
        u = safe_int(row.get("final_update_id"))
        pu = safe_int(row.get("previous_final_update_id"))
        ts = safe_int(row.get("ts_recv_ns"))
        if U is None or u is None or pu is None:
            self.valid = False
            self.sequence_gap_count += 1
            return "MISSING_IDS"
        if self.bridge_pending:
            L = self.seed_last_u
            if L is None:
                self.valid = False
                return "NO_SEED_ID"
            if u < L:
                return "STALE_DIFF"
            if not (pu == L or U <= L <= u):
                self.valid = False
                self.bridge_pending = False
                self.sequence_gap_count += 1
                return "SEED_BRIDGE_FAILED"
            self.apply_levels(row.get("bid_updates") or [], row.get("ask_updates") or [])
            self.previous_u = u
            self.bridge_pending = False
        else:
            if self.previous_u is None or pu != self.previous_u:
                self.valid = False
                self.sequence_gap_count += 1
                return "PU_GAP"
            self.apply_levels(row.get("bid_updates") or [], row.get("ask_updates") or [])
            self.previous_u = u
        self.last_state_ts_ns = ts
        if not self.top_valid():
            self.valid = False
            self.crossed_after_diff_count += 1
            return "CROSSED_AFTER_DIFF"
        return "APPLIED"


def iter_parquet_rows(paths: Sequence[Path], columns: Sequence[str], symbol: str, batch_size: int) -> Iterator[dict[str, Any]]:
    _, _, _, pq = require_data_stack()
    for path_idx, path in enumerate(sorted(paths)):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        use = [c for c in columns if c in available]
        missing = [c for c in columns if c not in available]
        required = {"ts_recv_ns", "symbol"}
        if required.intersection(missing):
            raise RuntimeError(f"{path}: missing required columns {sorted(required.intersection(missing))}")
        record_base = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=use):
            names = batch.schema.names
            py = {name: batch.column(i).to_pylist() for i, name in enumerate(names)}
            n = batch.num_rows
            for i in range(n):
                row = {name: py[name][i] for name in names}
                if str(row.get("symbol", "")) != symbol:
                    continue
                row["_path_index"] = path_idx
                row["_source_record_number"] = record_base + i
                yield row
            record_base += n


def merge_family(paths: Sequence[Path], columns: Sequence[str], symbol: str, batch_size: int, priority: int, kind: str) -> Iterator[dict[str, Any]]:
    iters = [iter_parquet_rows((p,), columns, symbol, batch_size) for p in sorted(paths)]
    heap: list[tuple[int, int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for idx, it in enumerate(iters):
        try:
            row = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(row["ts_recv_ns"]), idx, int(row.get("_source_record_number", 0)), row, it))
    while heap:
        _, idx, _, row, it = heapq.heappop(heap)
        row["_priority"] = priority
        row["_kind"] = kind
        yield row
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(nxt["ts_recv_ns"]), idx, int(nxt.get("_source_record_number", 0)), nxt, it))


def merged_events(snapshot_files: Sequence[Path], depth_files: Sequence[Path], symbol: str) -> Iterator[dict[str, Any]]:
    snap_cols = ["ts_recv_ns", "symbol", "last_update_id", "bids", "asks"]
    depth_cols = [
        "ts_recv_ns", "symbol", "first_update_id", "final_update_id", "previous_final_update_id",
        "bid_updates", "ask_updates",
    ]
    snaps = merge_family(snapshot_files, snap_cols, symbol, 8, 0, "SNAPSHOT")
    depth = merge_family(depth_files, depth_cols, symbol, 2048, 1, "DEPTH")
    heap: list[tuple[int, int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for idx, it in enumerate((snaps, depth)):
        try:
            row = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(row["ts_recv_ns"]), int(row["_priority"]), idx, row, it))
    while heap:
        _, _, idx, row, it = heapq.heappop(heap)
        yield row
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(nxt["ts_recv_ns"]), int(nxt["_priority"]), idx, nxt, it))


def signed_bps(a: float | None, b: float | None) -> float:
    if a is None or b is None or a <= 0 or b <= 0:
        return math.nan
    return (a / b - 1.0) * 10_000.0


def process_day(canonical_root: Path, output_root: Path, report_root: Path, d: date, symbol: str, example_limit: int) -> dict[str, Any]:
    _, pd, pa, pq = require_data_stack()
    snapshots = discover_family_files(canonical_root, "book_snapshots", d)
    depth = discover_family_files(canonical_root, "depth_updates", d)
    if not snapshots or not depth:
        raise RuntimeError(f"{d}: requires both book_snapshots and depth_updates")

    detail_rows: list[dict[str, Any]] = []
    shadow = ShadowBook()
    last_depth: dict[str, Any] | None = None
    pending_snapshots: list[int] = []
    event_index = 0

    for ev in merged_events(snapshots, depth, symbol):
        event_index += 1
        kind = ev["_kind"]
        ts = int(ev["ts_recv_ns"])
        if kind == "SNAPSHOT":
            sem = snapshot_semantics(ev)
            sbid, sask = shadow.top() if shadow.valid else (None, None)
            smid = midpoint(sbid, sask)
            nmid = sem.get("normal_mid")
            row = {
                "receive_date": d.isoformat(),
                "snapshot_index": len(detail_rows),
                "snapshot_ts_recv_ns": ts,
                "snapshot_time": datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).isoformat(),
                "snapshot_last_update_id": safe_int(ev.get("last_update_id")),
                **sem,
                "previous_diff_ts_recv_ns": safe_int(last_depth.get("ts_recv_ns")) if last_depth else None,
                "previous_diff_latency_ms": (ts - int(last_depth["ts_recv_ns"])) / 1e6 if last_depth else math.nan,
                "previous_diff_U": safe_int(last_depth.get("first_update_id")) if last_depth else None,
                "previous_diff_u": safe_int(last_depth.get("final_update_id")) if last_depth else None,
                "previous_diff_pu": safe_int(last_depth.get("previous_final_update_id")) if last_depth else None,
                "next_diff_ts_recv_ns": None,
                "next_diff_latency_ms": math.nan,
                "next_diff_U": None,
                "next_diff_u": None,
                "next_diff_pu": None,
                "bridge_class": "PENDING",
                "shadow_valid_before_snapshot": bool(shadow.valid),
                "shadow_best_bid": sbid,
                "shadow_best_ask": sask,
                "shadow_mid": smid,
                "snapshot_vs_shadow_mid_bps": signed_bps(nmid, smid),
                "snapshot_bid_vs_shadow_bid_bps": signed_bps(sem.get("normal_best_bid"), sbid),
                "snapshot_ask_vs_shadow_ask_bps": signed_bps(sem.get("normal_best_ask"), sask),
                "shadow_seed_age_ms": (ts - shadow.seed_ts_ns) / 1e6 if shadow.valid and shadow.seed_ts_ns is not None else math.nan,
                "shadow_state_age_ms": (ts - shadow.last_state_ts_ns) / 1e6 if shadow.valid and shadow.last_state_ts_ns is not None else math.nan,
            }
            detail_rows.append(row)
            pending_snapshots.append(len(detail_rows) - 1)

            # Seed only if the shadow is currently invalid. Valid snapshots are
            # controls, not mandatory resets, while a continuous shadow chain exists.
            if not shadow.valid and sem["normal_status"] == "VALID":
                shadow.seed(ev, sem)
        else:
            # This is the first post-snapshot depth for every snapshot pending
            # since the previous depth message.
            if pending_snapshots:
                for idx in pending_snapshots:
                    r = detail_rows[idx]
                    r["next_diff_ts_recv_ns"] = ts
                    r["next_diff_latency_ms"] = (ts - int(r["snapshot_ts_recv_ns"])) / 1e6
                    r["next_diff_U"] = safe_int(ev.get("first_update_id"))
                    r["next_diff_u"] = safe_int(ev.get("final_update_id"))
                    r["next_diff_pu"] = safe_int(ev.get("previous_final_update_id"))
                    r["bridge_class"] = bridge_classification(r.get("snapshot_last_update_id"), ev)
                pending_snapshots.clear()

            result = shadow.apply_depth(ev)
            if not shadow.valid and result in {"SEED_BRIDGE_FAILED", "PU_GAP", "CROSSED_AFTER_DIFF", "MISSING_IDS"}:
                # Remain invalid until the next normal-valid snapshot seeds it.
                pass
            last_depth = ev

    # No later diff for snapshots at end of day.
    for idx in pending_snapshots:
        detail_rows[idx]["bridge_class"] = "NO_NEXT_DIFF"

    frame = pd.DataFrame(detail_rows)
    if frame.empty:
        raise RuntimeError(f"{d}: no snapshots found for {symbol}")

    out_dir = output_root / f"symbol={symbol}" / f"date={d.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"snapshot-semantics-{d.isoformat()}.parquet"
    tmp = out_dir / f".snap-{uuid.uuid4().hex[:8]}.tmp"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, detail_path)

    status_counts = frame["normal_status"].value_counts(dropna=False).to_dict()
    swapped_counts = frame["swapped_status"].value_counts(dropna=False).to_dict()
    first_counts = frame["first_listed_status"].value_counts(dropna=False).to_dict()
    bridge_counts = frame["bridge_class"].value_counts(dropna=False).to_dict()

    summary = {
        "receive_date": d.isoformat(),
        "snapshots": int(len(frame)),
        "normal_valid": int(status_counts.get("VALID", 0)),
        "normal_valid_rate": float((frame["normal_status"] == "VALID").mean()),
        "normal_crossed": int(status_counts.get("CROSSED", 0)),
        "swapped_valid": int(swapped_counts.get("VALID", 0)),
        "swapped_valid_rate": float((frame["swapped_status"] == "VALID").mean()),
        "first_listed_valid": int(first_counts.get("VALID", 0)),
        "first_listed_valid_rate": float((frame["first_listed_status"] == "VALID").mean()),
        "local_checkpoint_compatible": int(bridge_counts.get("LOCAL_CHECKPOINT_COMPATIBLE", 0)),
        "rest_bridge_compatible": int(bridge_counts.get("REST_BRIDGE_COMPATIBLE", 0)),
        "no_sequence_bridge": int(bridge_counts.get("NO_SEQUENCE_BRIDGE", 0)),
        "next_diff_stale": int(bridge_counts.get("NEXT_DIFF_STALE", 0)),
        "shadow_comparable_snapshots": int(frame["shadow_valid_before_snapshot"].sum()),
        "shadow_sequence_gaps": int(shadow.sequence_gap_count),
        "shadow_crossed_after_diff": int(shadow.crossed_after_diff_count),
        "detail_path": str(detail_path),
    }

    # Aggregated report rows.
    top_rows = [
        {"receive_date": d.isoformat(), "interpretation": "NORMAL", "status": str(k), "count": int(v), "share": float(v / len(frame))}
        for k, v in sorted(status_counts.items())
    ]
    top_rows += [
        {"receive_date": d.isoformat(), "interpretation": "SWAPPED", "status": str(k), "count": int(v), "share": float(v / len(frame))}
        for k, v in sorted(swapped_counts.items())
    ]
    top_rows += [
        {"receive_date": d.isoformat(), "interpretation": "FIRST_LISTED", "status": str(k), "count": int(v), "share": float(v / len(frame))}
        for k, v in sorted(first_counts.items())
    ]

    bridge_rows = [
        {"receive_date": d.isoformat(), "bridge_class": str(k), "count": int(v), "share": float(v / len(frame))}
        for k, v in sorted(bridge_counts.items())
    ]

    profile_rows: list[dict[str, Any]] = []
    for label, mask in (("NORMAL_VALID", frame["normal_status"] == "VALID"), ("NORMAL_INVALID", frame["normal_status"] != "VALID")):
        g = frame.loc[mask]
        if g.empty:
            continue
        def med(c: str) -> float:
            return float(pd.to_numeric(g[c], errors="coerce").median())
        profile_rows.append({
            "receive_date": d.isoformat(),
            "group": label,
            "snapshots": int(len(g)),
            "median_bid_raw_count": med("bid_raw_count"),
            "median_ask_raw_count": med("ask_raw_count"),
            "median_bid_usable_count": med("bid_usable_count"),
            "median_ask_usable_count": med("ask_usable_count"),
            "median_bid_descending_fraction": med("bid_descending_fraction"),
            "median_ask_ascending_fraction": med("ask_ascending_fraction"),
            "median_cross_bps": med("normal_cross_bps"),
            "first_listed_valid_share": float((g["first_listed_status"] == "VALID").mean()),
            "swapped_valid_share": float((g["swapped_status"] == "VALID").mean()),
            "local_checkpoint_share": float((g["bridge_class"] == "LOCAL_CHECKPOINT_COMPATIBLE").mean()),
            "shadow_comparable_share": float(g["shadow_valid_before_snapshot"].mean()),
            "median_snapshot_vs_shadow_mid_bps": med("snapshot_vs_shadow_mid_bps"),
        })

    compare = frame.loc[frame["shadow_valid_before_snapshot"]].copy()
    compare_rows: list[dict[str, Any]] = []
    if not compare.empty:
        for status, g in compare.groupby("normal_status", dropna=False):
            compare_rows.append({
                "receive_date": d.isoformat(),
                "normal_status": str(status),
                "snapshots": int(len(g)),
                "median_abs_snapshot_vs_shadow_mid_bps": float(pd.to_numeric(g["snapshot_vs_shadow_mid_bps"], errors="coerce").abs().median()),
                "p95_abs_snapshot_vs_shadow_mid_bps": float(pd.to_numeric(g["snapshot_vs_shadow_mid_bps"], errors="coerce").abs().quantile(0.95)),
                "median_abs_bid_vs_shadow_bps": float(pd.to_numeric(g["snapshot_bid_vs_shadow_bid_bps"], errors="coerce").abs().median()),
                "median_abs_ask_vs_shadow_bps": float(pd.to_numeric(g["snapshot_ask_vs_shadow_ask_bps"], errors="coerce").abs().median()),
                "median_shadow_state_age_ms": float(pd.to_numeric(g["shadow_state_age_ms"], errors="coerce").median()),
            })

    # Examples prioritize crossed/empty and capture normal/swapped/first/shadow context.
    bad = frame.loc[frame["normal_status"] != "VALID"].copy()
    examples = []
    if not bad.empty:
        cols = [
            "receive_date", "snapshot_index", "snapshot_time", "snapshot_last_update_id",
            "normal_status", "normal_best_bid", "normal_best_ask", "normal_cross_bps",
            "swapped_status", "swapped_best_bid", "swapped_best_ask",
            "first_listed_status", "first_listed_bid", "first_listed_ask",
            "bid_raw_count", "ask_raw_count", "bid_descending_fraction", "ask_ascending_fraction",
            "crossed_bid_level_count", "crossed_ask_level_count", "bridge_class",
            "shadow_valid_before_snapshot", "shadow_best_bid", "shadow_best_ask",
            "snapshot_vs_shadow_mid_bps", "next_diff_latency_ms",
        ]
        examples = bad.loc[:, cols].head(example_limit).to_dict("records")

    return {
        "summary": summary,
        "top_rows": top_rows,
        "bridge_rows": bridge_rows,
        "profile_rows": profile_rows,
        "compare_rows": compare_rows,
        "examples": examples,
    }


def qtile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    x = (len(vals) - 1) * q
    lo = int(math.floor(x)); hi = int(math.ceil(x))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - x) + vals[hi] * (x - lo)


def build_findings(summaries: Sequence[Mapping[str, Any]], top_rows: Sequence[Mapping[str, Any]], profile_rows: Sequence[Mapping[str, Any]], bridge_rows: Sequence[Mapping[str, Any]]) -> str:
    total = sum(int(x["snapshots"]) for x in summaries)
    normal_valid = sum(int(x["normal_valid"]) for x in summaries)
    swapped_valid = sum(int(x["swapped_valid"]) for x in summaries)
    first_valid = sum(int(x["first_listed_valid"]) for x in summaries)
    local = sum(int(x["local_checkpoint_compatible"]) for x in summaries)
    rest = sum(int(x["rest_bridge_compatible"]) for x in summaries)
    no_bridge = sum(int(x["no_sequence_bridge"]) for x in summaries)
    lines = [
        "# Phase 6A1 — Depth Snapshot Semantics Diagnostic",
        "",
        f"Diagnostic version: `{DIAGNOSTIC_VERSION}`",
        "",
        "This is a read-only semantics diagnostic. It does not alter canonical data or the Phase-6A reconstruction contract.",
        "",
        "## Headline counts",
        "",
        f"- Snapshots inspected: **{total:,}**",
        f"- Normal interpretation valid: **{normal_valid:,} ({normal_valid/total:.2%})**" if total else "- Normal interpretation valid: n/a",
        f"- Swapped-side interpretation valid: **{swapped_valid:,} ({swapped_valid/total:.2%})**" if total else "- Swapped-side interpretation valid: n/a",
        f"- First-listed top valid: **{first_valid:,} ({first_valid/total:.2%})**" if total else "- First-listed top valid: n/a",
        f"- Next diff local-checkpoint compatible: **{local:,} ({local/total:.2%})**" if total else "- Local checkpoint: n/a",
        f"- Next diff REST-bridge compatible: **{rest:,} ({rest/total:.2%})**" if total else "- REST bridge: n/a",
        f"- Next diff with no sequence bridge: **{no_bridge:,} ({no_bridge/total:.2%})**" if total else "- No bridge: n/a",
        "",
        "## Interpretation guide",
        "",
        "- If `SWAPPED` validity is dramatically higher than `NORMAL`, side labels are likely reversed in the stored snapshot payload.",
        "- If `FIRST_LISTED` is usually valid while extrema-derived `NORMAL` is crossed, the arrays likely contain stale/partial levels rather than a coherent full-book checkpoint.",
        "- If suspect snapshots disagree sharply with a valid shadow book while the `pu -> u` diff chain remains continuous, snapshot content/semantics are the likely problem rather than the diff feed.",
        "- `NO_SEQUENCE_BRIDGE` is a sequence-semantics finding and must be kept separate from top-of-book validity.",
        "",
        "## Research contract",
        "",
        "No reconstruction rule should be changed merely to increase validity. The next Phase-6A patch should follow the interpretation supported by these diagnostics.",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    valid = {
        "bids": [{"price": "100", "quantity": "2"}, {"price": "99", "quantity": "3"}],
        "asks": [{"price": "101", "quantity": "2"}, {"price": "102", "quantity": "3"}],
    }
    s = snapshot_semantics(valid)
    assert s["normal_status"] == "VALID"
    assert s["normal_best_bid"] == 100 and s["normal_best_ask"] == 101
    assert s["bid_descending_fraction"] == 1.0 and s["ask_ascending_fraction"] == 1.0

    reversed_labels = {
        "bids": [{"price": "101", "quantity": "2"}, {"price": "102", "quantity": "3"}],
        "asks": [{"price": "100", "quantity": "2"}, {"price": "99", "quantity": "3"}],
    }
    r = snapshot_semantics(reversed_labels)
    assert r["normal_status"] == "CROSSED"
    assert r["swapped_status"] == "VALID"

    stale_deep = {
        "bids": [{"price": "100", "quantity": "1"}, {"price": "103", "quantity": "1"}],
        "asks": [{"price": "101", "quantity": "1"}, {"price": "98", "quantity": "1"}],
    }
    x = snapshot_semantics(stale_deep)
    assert x["normal_status"] == "CROSSED"
    assert x["first_listed_status"] == "VALID"

    assert bridge_classification(100, {"first_update_id": 101, "final_update_id": 101, "previous_final_update_id": 100}) == "LOCAL_CHECKPOINT_COMPATIBLE"
    assert bridge_classification(100, {"first_update_id": 99, "final_update_id": 101, "previous_final_update_id": 98}) == "REST_BRIDGE_COMPATIBLE"
    assert bridge_classification(100, {"first_update_id": 102, "final_update_id": 103, "previous_final_update_id": 101}) == "NO_SEQUENCE_BRIDGE"
    assert bridge_classification(100, {"first_update_id": 90, "final_update_id": 99, "previous_final_update_id": 89}) == "NEXT_DIFF_STALE"

    shadow = ShadowBook()
    snap = {"ts_recv_ns": 1_000_000_000, "last_update_id": 100, **valid}
    shadow.seed(snap, snapshot_semantics(snap))
    assert shadow.valid and shadow.bridge_pending
    result = shadow.apply_depth({
        "ts_recv_ns": 1_100_000_000,
        "first_update_id": 101, "final_update_id": 101, "previous_final_update_id": 100,
        "bid_updates": [{"price": "100", "quantity": "4"}], "ask_updates": [],
    })
    assert result == "APPLIED" and shadow.valid and not shadow.bridge_pending
    result = shadow.apply_depth({
        "ts_recv_ns": 1_200_000_000,
        "first_update_id": 102, "final_update_id": 102, "previous_final_update_id": 999,
        "bid_updates": [], "ask_updates": [],
    })
    assert result == "PU_GAP" and not shadow.valid
    print("SELF-TEST PASS")
    print("Validated precise snapshot top classification, swapped-side and first-listed hypotheses, sequence-bridge classes, and continuous shadow-book gap handling.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_SMOKE_DATE)
    ap.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_SMOKE_DATE)
    ap.add_argument("--example-limit", type=int, default=200)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be >= --start-date")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    days = list(date_range(args.start_date, args.end_date))
    print("Diagnosing BTCUSDT stored depth-snapshot semantics...")
    print(f"Diagnostic version : {DIAGNOSTIC_VERSION}")
    print(f"Symbol             : {args.symbol}")
    print(f"Canonical root     : {args.canonical_root}")
    print(f"Output root        : {args.output_root}")
    print(f"Report root        : {args.report_root}")
    print(f"Dates              : {len(days)} ({args.start_date} to {args.end_date})")
    print("Read-only          : YES — canonical data and Phase-6A outputs are not modified")

    all_summaries: list[dict[str, Any]] = []
    all_top: list[dict[str, Any]] = []
    all_bridge: list[dict[str, Any]] = []
    all_profile: list[dict[str, Any]] = []
    all_compare: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for i, d in enumerate(days, 1):
        try:
            result = process_day(args.canonical_root, args.output_root, args.report_root, d, args.symbol, args.example_limit)
            all_summaries.append(result["summary"])
            all_top.extend(result["top_rows"])
            all_bridge.extend(result["bridge_rows"])
            all_profile.extend(result["profile_rows"])
            all_compare.extend(result["compare_rows"])
            remaining = max(0, args.example_limit - len(all_examples))
            all_examples.extend(result["examples"][:remaining])
            s = result["summary"]
            print(
                f"[{i:02d}/{len(days):02d}] {d}: COMPLETE "
                f"({s['snapshots']:,} snapshots, normal-valid={s['normal_valid_rate']:.2%}, "
                f"swapped-valid={s['swapped_valid_rate']:.2%}, shadow={s['shadow_comparable_snapshots']:,})"
            )
        except Exception as exc:
            failures.append({"receive_date": d.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:02d}/{len(days):02d}] {d}: FAIL - {type(exc).__name__}: {exc}")

    write_csv(args.report_root / "snapshot_semantics_summary.csv", all_summaries)
    write_csv(args.report_root / "snapshot_top_status.csv", all_top)
    write_csv(args.report_root / "snapshot_interpretation_comparison.csv", all_top)
    write_csv(args.report_root / "snapshot_sequence_bridge.csv", all_bridge)
    write_csv(args.report_root / "snapshot_vs_surrounding_book.csv", all_compare)
    write_csv(args.report_root / "snapshot_valid_vs_invalid_profile.csv", all_profile)
    write_csv(args.report_root / "snapshot_failure_examples.csv", all_examples)
    write_csv(args.report_root / "snapshot_diagnostic_failures.csv", failures)

    findings = build_findings(all_summaries, all_top, all_profile, all_bridge)
    atomic_write_text(args.report_root / "snapshot_semantics_findings.md", findings)
    meta = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "symbol": args.symbol,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "dates_requested": len(days),
        "dates_completed": len(all_summaries),
        "dates_failed": len(failures),
        "snapshots": sum(int(x["snapshots"]) for x in all_summaries),
        "status": "COMPLETE" if not failures else "PARTIAL",
        "read_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.output_root / "metadata" / "latest_summary.json", meta)

    print("\nPhase-6A1 snapshot semantics diagnostic")
    print("---------------------------------------")
    print(f"Status             : {meta['status']}")
    print(f"Dates completed    : {meta['dates_completed']}/{meta['dates_requested']}")
    print(f"Snapshots inspected: {meta['snapshots']:,}")
    if all_summaries:
        total = meta["snapshots"]
        nv = sum(int(x["normal_valid"]) for x in all_summaries)
        sv = sum(int(x["swapped_valid"]) for x in all_summaries)
        fv = sum(int(x["first_listed_valid"]) for x in all_summaries)
        shadow = sum(int(x["shadow_comparable_snapshots"]) for x in all_summaries)
        print(f"Normal valid       : {nv:,} ({nv/total:.2%})")
        print(f"Swapped valid      : {sv:,} ({sv/total:.2%})")
        print(f"First-listed valid : {fv:,} ({fv/total:.2%})")
        print(f"Shadow comparable  : {shadow:,} ({shadow/total:.2%})")
    print(f"Findings           : {args.report_root / 'snapshot_semantics_findings.md'}")
    print("No reconstruction rule was changed by this diagnostic.")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
