#!/usr/bin/env python3
"""Build the Phase-6A BTCUSDT execution/data-quality foundation.

Purpose
-------
Phase 6A deliberately does *not* search for alpha. It establishes whether the
recent Stream perps order-book data are trustworthy enough for event research
and what a hypothetical taker would actually have paid to cross the book.

The builder consumes the canonical receive-time event archive produced by
``scripts/01_extract_all_market_events.py`` and reconstructs the Binance USD-M
local order book from stored snapshots plus diff-depth updates.

Core outputs
------------
1. A causal one-second execution-state dataset with:
   * book-validity / sequence-quality state,
   * best bid/ask/mid/spread,
   * visible quote depth inside 5/10/25 bps,
   * executable taker VWAP and crossing cost for several quote notionals,
   * contemporaneous trade-flow and forceOrder context.
2. An order-book integrity audit, including sequence gaps and recovery latency.
3. A stress-conditioned integrity audit (quality during high flow / high return
   / forceOrder seconds rather than only average quality).
4. A forceOrder semantics/profile audit. The size field is treated as a
   selected/censored observation and is never interpreted as total liquidation
   volume.

Important contracts
-------------------
* ``ts_recv_ns`` is the causal ordering clock.
* Stored ``book_snapshots`` are collector-local checkpoints, not raw Binance
  REST snapshots. A complete two-sided checkpoint may bootstrap/recover an
  invalid replay only if the first following depth update proves the exact
  local continuation ``pu == snapshot_last_update_id``.
* Once that handshake succeeds, the diff-depth stream is authoritative. Later
  periodic LocalBook snapshots are audit-only and never reset or invalidate a
  healthy reconstructed book.
* On a genuine ``pu != previous_u`` gap or a crossed reconstructed book, state
  becomes invalid until the next complete checkpoint plus exact local
  continuation. We never silently repair a gap.
* Diff quantities are absolute quantities. Quantity zero removes the level.
* Execution prices come only from visible reconstructed depth; insufficient
  visible depth is reported as an incomplete fill, not extrapolated.
* Replay acceleration is semantics-preserving: canonical nested depth strings are converted to numeric arrays in PyArrow and absolute updates are applied in a Numba-compiled integer-tick book.
* No trading fee is assumed by default. ``--taker-fee-bps`` can parameterize a
  user's own cost assumption without changing raw crossing-cost columns.
* forceOrder observed size is explicitly censored/selected. Per-second sums are
  retained only as descriptive sums of observed selected snapshots and are never
  interpreted or calibrated as true liquidation volume.

Default input
-------------
    data/canonical/market_events_v1

Expected canonical families
---------------------------
    perps/depth_updates/date=YYYY-MM-DD/*.parquet
    perps/book_snapshots/date=YYYY-MM-DD/*.parquet
    perps/trades/date=YYYY-MM-DD/*.parquet            (optional but recommended)
    perps/liquidations/date=YYYY-MM-DD/*.parquet      (optional)

Default output
--------------
    data/derived/perps/execution_foundation_v1/
        symbol=BTCUSDT/date=YYYY-MM-DD/part-YYYY-MM-DD.parquet
        metadata/build_state.sqlite
        metadata/latest_summary.json
        metadata/run_config.json

Reports
-------
    reports/perps_execution_foundation/
        execution_foundation_daily_quality.csv
        execution_foundation_sequence_gaps.csv
        execution_foundation_stress_quality.csv
        execution_foundation_execution_costs.csv
        execution_foundation_liquidation_profile.csv
        execution_foundation_liquidation_timing.csv
        execution_foundation_performance.csv
        execution_foundation_invariants.csv
        execution_foundation_findings.md

Typical usage
-------------
Smoke test one day first::

    python scripts/perps/research/06a_build_execution_foundation.py \
      --start-date 2026-06-12 --end-date 2026-06-12

Then run all available canonical days::

    python scripts/perps/research/06a_build_execution_foundation.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import heapq
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as _np

try:
    import numba as _numba  # type: ignore
    from numba import njit as _njit, types as _numba_types  # type: ignore
    from numba.typed import Dict as _NumbaDict  # type: ignore
    _NUMBA_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - exercised only on missing dependency
    _numba = None
    _njit = None
    _numba_types = None
    _NumbaDict = None
    _NUMBA_IMPORT_ERROR = _exc

BUILDER_VERSION = "1.0.4"
OUTPUT_SCHEMA_VERSION = 2
OUTPUT_SCHEMA_NAME = "perps_execution_foundation_v1"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_NOTIONALS = (1_000.0, 5_000.0, 10_000.0, 25_000.0)
DEPTH_BANDS_BPS = (5.0, 10.0, 25.0)
SECONDS_PER_DAY = 86_400

# A snapshot is considered too stale for an execution state if no valid diff or
# snapshot has been received for this long. The row is retained with book_valid
# false so downstream event studies can see the missingness explicitly.
DEFAULT_MAX_BOOK_AGE_MS = 2_000

# Stress buckets are descriptive audit strata, not causal trading thresholds.
STRESS_QUANTILES = (0.50, 0.90, 0.99)

# Performance-only tuning. These do not change research semantics.
PRICE_BLOCK_TARGET = 256
DEPTH_STREAM_BATCH_SIZE = 8192
SNAPSHOT_STREAM_BATCH_SIZE = 64
DEFAULT_PROGRESS_EVERY_DEPTH = 250_000
DEFAULT_PRICE_SCALE = 10.0  # BTCUSDT price ticks per USD (0.1 USDT tick); validated on every fast batch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def find_repo_root(script_path: Path) -> Path:
    p = script_path.resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "scripts").exists():
            return parent
    try:
        return p.parents[3]
    except Exception:
        return Path.cwd()


REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "data" / "canonical" / "market_events_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "derived" / "perps" / "execution_foundation_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "perps_execution_foundation"


def short_tmp(path: Path, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{prefix}-{uuid.uuid4().hex[:8]}.tmp"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = short_tmp(path, "tx")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def require_data_stack():
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Phase 6A requires numpy, pandas and pyarrow. Activate the project .venv first."
        ) from exc
    return np, pd, pa, pq


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def file_signature(paths: Sequence[Path]) -> str:
    rows = []
    for p in sorted(paths):
        st = p.stat()
        rows.append((str(p.resolve()), int(st.st_size), int(st.st_mtime_ns)))
    return sha256_text(compact_json(rows))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def bps_from_ratio(value: float, reference: float) -> float:
    if value <= 0 or reference <= 0:
        return math.nan
    return (value / reference - 1.0) * 10_000.0


def require_acceleration_stack() -> str:
    """Require the compiled replay dependency with an actionable error."""
    if _numba is None or _NumbaDict is None or _numba_types is None:
        raise SystemExit(
            "Phase 6A v1.0.4 requires Numba for the compiled depth replay.\n"
            "Activate the project .venv and install it with:\n"
            "  python -m pip install numba\n"
            f"Original import error: {_NUMBA_IMPORT_ERROR}"
        )
    return str(getattr(_numba, "__version__", "unknown"))


if _njit is not None:
    @_njit(cache=True)
    def _apply_tick_updates_kernel(levels, ticks, quantities, min_tick, max_tick, active_count):
        inserts = 0
        modifies = 0
        deletes = 0
        noops = 0
        for i in range(ticks.size):
            tick = int(ticks[i])
            q = float(quantities[i])
            if tick <= 0 or not math.isfinite(q) or q < 0.0:
                noops += 1
                continue
            exists = tick in levels
            if q == 0.0:
                if exists:
                    del levels[tick]
                    deletes += 1
                    active_count -= 1
                    if active_count == 0:
                        min_tick = 0
                        max_tick = 0
                    else:
                        if tick == min_tick:
                            t = min_tick + 1
                            while t <= max_tick and t not in levels:
                                t += 1
                            min_tick = t
                        if tick == max_tick:
                            t = max_tick - 1
                            while t >= min_tick and t not in levels:
                                t -= 1
                            max_tick = t
                else:
                    noops += 1
                continue
            if exists:
                levels[tick] = q
                modifies += 1
            else:
                levels[tick] = q
                inserts += 1
                active_count += 1
                if active_count == 1:
                    min_tick = tick
                    max_tick = tick
                else:
                    if tick < min_tick:
                        min_tick = tick
                    if tick > max_tick:
                        max_tick = tick
        return min_tick, max_tick, active_count, inserts, modifies, deletes, noops


    @_njit(cache=True)
    def _scan_tick_book_kernel(levels, is_bid, min_tick, max_tick, price_scale, mid, bands_bps, notionals):
        nb = bands_bps.size
        nn = notionals.size
        depths = _np.empty(nb, dtype=_np.float64)
        depths[:] = _np.nan
        vwaps = _np.empty(nn, dtype=_np.float64)
        vwaps[:] = _np.nan
        quote_done_out = _np.zeros(nn, dtype=_np.float64)
        base_done_out = _np.zeros(nn, dtype=_np.float64)
        complete_out = _np.zeros(nn, dtype=_np.uint8)
        if min_tick <= 0 or max_tick <= 0 or mid <= 0.0:
            return depths, vwaps, quote_done_out, base_done_out, complete_out

        boundaries = _np.empty(nb, dtype=_np.float64)
        for i in range(nb):
            if is_bid:
                boundaries[i] = mid * (1.0 - bands_bps[i] / 10000.0)
            else:
                boundaries[i] = mid * (1.0 + bands_bps[i] / 10000.0)

        band_i = 0
        notional_i = 0
        cumulative_quote = 0.0
        cumulative_base = 0.0
        tick = max_tick if is_bid else min_tick
        stop = min_tick if is_bid else max_tick
        step = -1 if is_bid else 1

        while (tick >= stop if is_bid else tick <= stop):
            if tick in levels:
                q = levels[tick]
                if q > 0.0:
                    p = tick / price_scale
                    while band_i < nb:
                        outside = p < boundaries[band_i] if is_bid else p > boundaries[band_i]
                        if not outside:
                            break
                        depths[band_i] = cumulative_quote
                        band_i += 1

                    previous_quote = cumulative_quote
                    previous_base = cumulative_base
                    level_quote = p * q
                    cumulative_quote += level_quote
                    cumulative_base += level_quote / p

                    while notional_i < nn:
                        target = notionals[notional_i]
                        tol = max(1e-9, target * 1e-12)
                        if cumulative_quote + tol < target:
                            break
                        if cumulative_quote >= target:
                            take_quote = max(0.0, target - previous_quote)
                            target_quote = target
                            target_base = previous_base + take_quote / p
                        else:
                            target_quote = cumulative_quote
                            target_base = cumulative_base
                        vwaps[notional_i] = target_quote / target_base if target_base > 0.0 else _np.nan
                        quote_done_out[notional_i] = target_quote
                        base_done_out[notional_i] = target_base
                        complete_out[notional_i] = 1
                        notional_i += 1

                    if band_i == nb and notional_i == nn:
                        break
            tick += step

        while band_i < nb:
            depths[band_i] = cumulative_quote
            band_i += 1
        while notional_i < nn:
            target = notionals[notional_i]
            tol = max(1e-9, target * 1e-12)
            remaining = max(0.0, target - cumulative_quote)
            done = remaining <= tol
            vwaps[notional_i] = cumulative_quote / cumulative_base if cumulative_base > 0.0 else _np.nan
            quote_done_out[notional_i] = cumulative_quote
            base_done_out[notional_i] = cumulative_base
            complete_out[notional_i] = 1 if done else 0
            notional_i += 1
        return depths, vwaps, quote_done_out, base_done_out, complete_out
else:
    def _apply_tick_updates_kernel(*args, **kwargs):  # type: ignore
        require_acceleration_stack()
        raise AssertionError("unreachable")

    def _scan_tick_book_kernel(*args, **kwargs):  # type: ignore
        require_acceleration_stack()
        raise AssertionError("unreachable")


@dataclass
class FastDepthStats:
    arrow_batches: int = 0
    rows_decoded: int = 0
    levels_decoded: int = 0
    arrow_decode_seconds: float = 0.0


@dataclass(frozen=True)
class DayFiles:
    day: date
    depth: tuple[Path, ...]
    snapshots: tuple[Path, ...]
    trades: tuple[Path, ...]
    liquidations: tuple[Path, ...]

    @property
    def required_ok(self) -> bool:
        return bool(self.depth and self.snapshots)

    @property
    def all_files(self) -> tuple[Path, ...]:
        return self.depth + self.snapshots + self.trades + self.liquidations


@dataclass
class GapRecord:
    receive_date: str
    ts_recv_ns: int
    reason: str
    previous_u: int | None
    pu: int | None
    first_update_id: int | None
    final_update_id: int | None
    snapshot_last_update_id: int | None


@dataclass
class Check:
    scope: str
    check: str
    status: str
    observed: Any
    expected: Any
    detail: str = ""


class ChunkedSortedPrices:
    """Ascending unique prices in small sorted blocks.

    Python list insertion/removal shifts the entire tail of a monolithic list.
    Depth replay performs millions of such mutations, so v1.0.3 bounds those
    shifts to a small block while retaining deterministic sorted iteration and
    no third-party dependency.
    """

    def __init__(self, target: int = PRICE_BLOCK_TARGET):
        self.target = max(32, int(target))
        self.blocks: list[list[float]] = []
        self.maxes: list[float] = []
        self.length = 0

    def __len__(self) -> int:
        return self.length

    def clear(self) -> None:
        self.blocks.clear()
        self.maxes.clear()
        self.length = 0

    def build(self, values: Iterable[float]) -> None:
        vals = sorted(set(values))
        self.clear()
        if not vals:
            return
        step = self.target
        self.blocks = [vals[i:i + step] for i in range(0, len(vals), step)]
        self.maxes = [b[-1] for b in self.blocks]
        self.length = len(vals)

    def _block_index(self, value: float) -> int:
        if not self.blocks:
            return -1
        i = bisect.bisect_left(self.maxes, value)
        if i >= len(self.blocks):
            i = len(self.blocks) - 1
        return i

    def add(self, value: float) -> bool:
        if not self.blocks:
            self.blocks = [[value]]
            self.maxes = [value]
            self.length = 1
            return True
        i = self._block_index(value)
        block = self.blocks[i]
        j = bisect.bisect_left(block, value)
        if j < len(block) and block[j] == value:
            return False
        block.insert(j, value)
        self.length += 1
        self.maxes[i] = block[-1]
        if len(block) > self.target * 2:
            cut = len(block) // 2
            right = block[cut:]
            del block[cut:]
            self.blocks.insert(i + 1, right)
            self.maxes[i] = block[-1]
            self.maxes.insert(i + 1, right[-1])
        return True

    def discard(self, value: float) -> bool:
        if not self.blocks:
            return False
        i = bisect.bisect_left(self.maxes, value)
        if i >= len(self.blocks):
            return False
        block = self.blocks[i]
        j = bisect.bisect_left(block, value)
        if j >= len(block) or block[j] != value:
            return False
        block.pop(j)
        self.length -= 1
        if not block:
            self.blocks.pop(i)
            self.maxes.pop(i)
            return True
        self.maxes[i] = block[-1]
        # Keep blocks reasonably full so long replays do not fragment into many
        # tiny lists. Merging preserves exact sorted order.
        if len(block) < max(8, self.target // 4) and len(self.blocks) > 1:
            if i + 1 < len(self.blocks) and len(block) + len(self.blocks[i + 1]) <= self.target * 2:
                block.extend(self.blocks.pop(i + 1))
                self.maxes.pop(i + 1)
                self.maxes[i] = block[-1]
            elif i > 0 and len(self.blocks[i - 1]) + len(block) <= self.target * 2:
                left = self.blocks[i - 1]
                left.extend(block)
                self.blocks.pop(i)
                self.maxes.pop(i)
                self.maxes[i - 1] = left[-1]
        return True

    def iter_asc(self) -> Iterator[float]:
        for block in self.blocks:
            yield from block

    def iter_desc(self) -> Iterator[float]:
        for block in reversed(self.blocks):
            yield from reversed(block)


class LegacyBookSide:
    """v1.0.3 Python reference implementation retained for parity self-tests."""

    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.prices = ChunkedSortedPrices()
        self.qty: dict[float, float] = {}

    def clear(self) -> None:
        self.prices.clear()
        self.qty.clear()

    def reset(self, levels: Sequence[Any]) -> None:
        self.clear()
        for level in levels or []:
            if isinstance(level, Mapping):
                p = safe_float(level.get("price"))
                q = safe_float(level.get("quantity"))
            else:
                try:
                    p = safe_float(level[0])
                    q = safe_float(level[1])
                except Exception:
                    continue
            if p is None or q is None or p <= 0 or q <= 0:
                continue
            self.qty[p] = q
        self.prices.build(self.qty.keys())

    def update(self, price: float, quantity: float) -> int:
        """Apply an absolute update; return 1 insert, 2 modify, 3 delete, 0 no-op."""
        if not math.isfinite(price) or price <= 0 or not math.isfinite(quantity) or quantity < 0:
            return 0
        exists = price in self.qty
        if quantity == 0:
            if exists:
                del self.qty[price]
                self.prices.discard(price)
                return 3
            return 0
        if exists:
            self.qty[price] = quantity
            return 2
        self.prices.add(price)
        self.qty[price] = quantity
        return 1

    def best(self) -> tuple[float | None, float | None]:
        if len(self.prices) == 0:
            return None, None
        if self.is_bid:
            block = self.prices.blocks[-1]
            p = block[-1]
        else:
            block = self.prices.blocks[0]
            p = block[0]
        return p, self.qty.get(p)

    def iter_best(self) -> Iterator[tuple[float, float]]:
        iterable = self.prices.iter_desc() if self.is_bid else self.prices.iter_asc()
        for p in iterable:
            q = self.qty.get(p)
            if q is not None and q > 0:
                yield p, q

    def scan_depth_and_execution(
        self,
        mid: float,
        bands_bps: Sequence[float],
        notionals: Sequence[float],
    ) -> tuple[dict[float, float], dict[float, tuple[float, float, float, bool]]]:
        """Compute all bands/notionals with one cumulative ordered traversal.

        Band boundaries and notional targets advance monotonically as we move
        away from the touch, avoiding the old repeated walks and avoiding an
        inner loop over every target at every level.
        """
        bands = tuple(sorted(float(b) for b in bands_bps))
        ns = tuple(sorted(float(n) for n in notionals))
        depths: dict[float, float] = {}
        execution: dict[float, tuple[float, float, float, bool]] = {}
        if mid <= 0:
            return ({b: math.nan for b in bands}, execution)

        if self.is_bid:
            boundaries = [mid * (1.0 - b / 10_000.0) for b in bands]
            outside = lambda price, boundary: price < boundary
        else:
            boundaries = [mid * (1.0 + b / 10_000.0) for b in bands]
            outside = lambda price, boundary: price > boundary

        band_i = 0
        notional_i = 0
        cumulative_quote = 0.0
        cumulative_base = 0.0
        tolerances = [max(1e-9, n * 1e-12) for n in ns]

        for p, q in self.iter_best():
            while band_i < len(bands) and outside(p, boundaries[band_i]):
                depths[bands[band_i]] = cumulative_quote
                band_i += 1

            previous_quote = cumulative_quote
            previous_base = cumulative_base
            level_quote = p * q
            cumulative_quote += level_quote
            # Match the legacy arithmetic rather than simplifying to +q.
            cumulative_base += level_quote / p

            while notional_i < len(ns):
                target = ns[notional_i]
                tol = tolerances[notional_i]
                if cumulative_quote + tol < target:
                    break
                if cumulative_quote >= target:
                    take_quote = max(0.0, target - previous_quote)
                    target_quote = target
                    target_base = previous_base + take_quote / p
                else:
                    # Legacy completion tolerance can mark a tiny residual as
                    # complete even when visible cumulative quote is just shy.
                    target_quote = cumulative_quote
                    target_base = cumulative_base
                vwap = target_quote / target_base if target_base > 0 else math.nan
                execution[target] = (vwap, target_quote, target_base, True)
                notional_i += 1

            if band_i == len(bands) and notional_i == len(ns):
                break

        while band_i < len(bands):
            depths[bands[band_i]] = cumulative_quote
            band_i += 1

        while notional_i < len(ns):
            target = ns[notional_i]
            tol = tolerances[notional_i]
            remaining = max(0.0, target - cumulative_quote)
            done = remaining <= tol
            vwap = cumulative_quote / cumulative_base if cumulative_base > 0 else math.nan
            execution[target] = (vwap, cumulative_quote, cumulative_base, bool(done))
            notional_i += 1

        return depths, execution


class CompiledBookSide:
    """Numba-backed absolute-quantity book keyed by integer exchange price ticks."""

    def __init__(self, is_bid: bool, price_scale: float = DEFAULT_PRICE_SCALE):
        require_acceleration_stack()
        self.is_bid = bool(is_bid)
        self.price_scale = float(price_scale)
        if not math.isfinite(self.price_scale) or self.price_scale <= 0:
            raise ValueError("price_scale must be positive and finite")
        self.qty = _NumbaDict.empty(  # type: ignore[union-attr]
            key_type=_numba_types.int64, value_type=_numba_types.float64  # type: ignore[union-attr]
        )
        self.min_tick = 0
        self.max_tick = 0
        self.active_count = 0

    def __len__(self) -> int:
        return int(self.active_count)

    def clear(self) -> None:
        self.qty = _NumbaDict.empty(  # type: ignore[union-attr]
            key_type=_numba_types.int64, value_type=_numba_types.float64  # type: ignore[union-attr]
        )
        self.min_tick = 0
        self.max_tick = 0
        self.active_count = 0

    def _price_to_tick(self, price: float) -> int:
        scaled = float(price) * self.price_scale
        tick = int(round(scaled))
        if abs(scaled - tick) > 1e-6:
            raise RuntimeError(
                f"Price {price!r} is not aligned to --price-scale {self.price_scale:g}. "
                "Use a finer scale (for example --price-scale 100)."
            )
        return tick

    def reset(self, levels: Sequence[Any]) -> None:
        self.clear()
        ticks: list[int] = []
        qtys: list[float] = []
        for level in levels or []:
            if isinstance(level, Mapping):
                p = safe_float(level.get("price"))
                q = safe_float(level.get("quantity"))
            else:
                try:
                    p = safe_float(level[0])
                    q = safe_float(level[1])
                except Exception:
                    continue
            if p is None or q is None or p <= 0 or q <= 0:
                continue
            ticks.append(self._price_to_tick(p))
            qtys.append(q)
        if ticks:
            self.update_arrays(_np.asarray(ticks, dtype=_np.int64), _np.asarray(qtys, dtype=_np.float64))

    def update_arrays(self, ticks, quantities) -> tuple[int, int, int, int]:
        if len(ticks) == 0:
            return 0, 0, 0, 0
        (self.min_tick, self.max_tick, self.active_count,
         inserts, modifies, deletes, noops) = _apply_tick_updates_kernel(
            self.qty,
            _np.asarray(ticks, dtype=_np.int64),
            _np.asarray(quantities, dtype=_np.float64),
            int(self.min_tick),
            int(self.max_tick),
            int(self.active_count),
        )
        return int(inserts), int(modifies), int(deletes), int(noops)

    def update(self, price: float, quantity: float) -> int:
        tick = self._price_to_tick(price)
        ins, mod, dele, noop = self.update_arrays(
            _np.asarray([tick], dtype=_np.int64), _np.asarray([quantity], dtype=_np.float64)
        )
        if ins:
            return 1
        if mod:
            return 2
        if dele:
            return 3
        return 0

    def best(self) -> tuple[float | None, float | None]:
        if self.active_count <= 0:
            return None, None
        tick = int(self.max_tick if self.is_bid else self.min_tick)
        try:
            q = float(self.qty[tick])
        except KeyError:
            return None, None
        return tick / self.price_scale, q

    def iter_best(self) -> Iterator[tuple[float, float]]:
        if self.active_count <= 0:
            return
        tick = int(self.max_tick if self.is_bid else self.min_tick)
        stop = int(self.min_tick if self.is_bid else self.max_tick)
        step = -1 if self.is_bid else 1
        while tick >= stop if self.is_bid else tick <= stop:
            if tick in self.qty:
                q = float(self.qty[tick])
                if q > 0:
                    yield tick / self.price_scale, q
            tick += step

    def scan_depth_and_execution(
        self,
        mid: float,
        bands_bps: Sequence[float],
        notionals: Sequence[float],
    ) -> tuple[dict[float, float], dict[float, tuple[float, float, float, bool]]]:
        bands = _np.asarray(tuple(sorted(float(b) for b in bands_bps)), dtype=_np.float64)
        ns = _np.asarray(tuple(sorted(float(n) for n in notionals)), dtype=_np.float64)
        depths_a, vwaps, quotes, bases, complete = _scan_tick_book_kernel(
            self.qty,
            self.is_bid,
            int(self.min_tick),
            int(self.max_tick),
            float(self.price_scale),
            float(mid),
            bands,
            ns,
        )
        depths = {float(bands[i]): float(depths_a[i]) for i in range(len(bands))}
        execution = {
            float(ns[i]): (float(vwaps[i]), float(quotes[i]), float(bases[i]), bool(complete[i]))
            for i in range(len(ns))
        }
        return depths, execution


def warm_compiled_kernels(price_scale: float) -> float:
    """JIT-compile the tiny replay/metric kernels before day timing begins."""
    t0 = time.perf_counter()
    side = CompiledBookSide(False, price_scale)
    ticks = _np.asarray([1000, 1001, 1002], dtype=_np.int64)
    qty = _np.asarray([1.0, 2.0, 3.0], dtype=_np.float64)
    side.update_arrays(ticks, qty)
    side.scan_depth_and_execution(100.05, (5.0, 10.0, 25.0), (10.0, 25.0))
    return time.perf_counter() - t0


class LocalBook:
    """Historical replay book: complete local checkpoint -> authoritative diffs.

    Stored snapshots are collector-local observations. They are used only to
    bootstrap/recover an invalid replay. Once diff continuity is established,
    all later snapshots are audit-only and cannot mutate healthy state.
    """

    def __init__(self, price_scale: float = DEFAULT_PRICE_SCALE):
        self.price_scale = float(price_scale)
        self.bids = CompiledBookSide(True, self.price_scale)
        self.asks = CompiledBookSide(False, self.price_scale)
        self.valid = False
        self.bridge_pending = False
        self.snapshot_last_update_id: int | None = None
        self.previous_u: int | None = None
        self.last_state_ts_ns: int | None = None
        self.last_snapshot_ts_ns: int | None = None
        self.invalid_since_ns: int | None = None
        self.last_gap_ts_ns: int | None = None

        self.snapshot_count = 0
        self.complete_checkpoint_count = 0
        self.bid_only_checkpoint_count = 0
        self.ask_only_checkpoint_count = 0
        self.empty_checkpoint_count = 0
        self.invalid_checkpoint_count = 0
        self.audit_only_checkpoint_count = 0
        self.complete_checkpoint_match_count = 0
        self.complete_checkpoint_mismatch_count = 0

        self.bootstrap_candidate_count = 0
        self.bootstrap_count = 0
        self.recovery_count = 0
        self.bootstrap_bridge_failure_count = 0
        self.first_bootstrap_ts_ns: int | None = None
        self.last_bootstrap_ts_ns: int | None = None
        self.bootstrap_source = "NONE"

        self.depth_count = 0
        self.stale_depth_count = 0
        self.ignored_invalid_depth_count = 0
        self.actual_sequence_gap_count = 0
        self.crossed_count = 0
        self.last_bridge_mode = "NONE"
        self.recovery_latencies_ms: list[float] = []

        # Performance diagnostics for authoritative diff replay.
        self.price_insert_count = 0
        self.price_modify_count = 0
        self.price_delete_count = 0
        self.price_noop_count = 0
        self.bid_update_count = 0
        self.ask_update_count = 0
        self.max_bid_levels = 0
        self.max_ask_levels = 0

    @staticmethod
    def _top_ok_sides(bids: CompiledBookSide, asks: CompiledBookSide) -> bool:
        bid, _ = bids.best()
        ask, _ = asks.best()
        return bool(bid is not None and ask is not None and bid > 0 and ask > 0 and bid < ask)

    def _top_ok(self) -> bool:
        return self._top_ok_sides(self.bids, self.asks)

    @staticmethod
    def _snapshot_side_top(levels: Sequence[Any], is_bid: bool) -> tuple[float | None, float | None]:
        """Return best positive level without building/sorting a full book side."""
        best_p: float | None = None
        best_q: float | None = None
        for level in levels or []:
            if isinstance(level, Mapping):
                p = safe_float(level.get("price"))
                q = safe_float(level.get("quantity"))
            else:
                try:
                    p = safe_float(level[0])
                    q = safe_float(level[1])
                except Exception:
                    continue
            if p is None or q is None or p <= 0 or q <= 0:
                continue
            if best_p is None or (p > best_p if is_bid else p < best_p):
                best_p, best_q = p, q
        return best_p, best_q

    def _classify_snapshot_light(
        self, row: Mapping[str, Any]
    ) -> tuple[str, tuple[float | None, float | None], tuple[float | None, float | None], int | None]:
        bid_top = self._snapshot_side_top(row.get("bids") or [], True)
        ask_top = self._snapshot_side_top(row.get("asks") or [], False)
        last_u = row.get("last_update_id")
        try:
            last_u_i = int(last_u) if last_u is not None else None
        except Exception:
            last_u_i = None
        bid, _ = bid_top
        ask, _ = ask_top
        if bid is None and ask is None:
            status = "EMPTY_BOTH"
        elif bid is not None and ask is None:
            status = "BID_ONLY"
        elif ask is not None and bid is None:
            status = "ASK_ONLY"
        elif bid is not None and ask is not None and bid < ask and last_u_i is not None:
            status = "COMPLETE"
        else:
            status = "INVALID_TWO_SIDED"
        return status, bid_top, ask_top, last_u_i

    def invalidate(self, ts_ns: int, reason: str) -> None:
        if self.valid or self.bridge_pending or self.invalid_since_ns is None:
            self.invalid_since_ns = ts_ns
        self.valid = False
        self.bridge_pending = False
        self.snapshot_last_update_id = None
        self.previous_u = None
        self.last_bridge_mode = "INVALID"
        self.last_gap_ts_ns = ts_ns
        if reason == "PU_SEQUENCE_GAP":
            self.actual_sequence_gap_count += 1
        elif reason == "CROSSED_BOOK":
            self.crossed_count += 1

    def apply_snapshot(self, row: Mapping[str, Any]) -> None:
        """Audit every local checkpoint; mutate state only while invalid.

        A healthy diff replay is never reset by a periodic LocalBook snapshot.
        If replay is invalid, only a complete sane two-sided checkpoint with a
        last_update_id becomes a bootstrap candidate.  The candidate is not
        considered sequence-valid until the next diff proves pu == last_u.
        """
        ts = int(row["ts_recv_ns"])
        self.snapshot_count += 1
        self.last_snapshot_ts_ns = ts
        status, snap_bid_top, snap_ask_top, last_u_i = self._classify_snapshot_light(row)

        if status == "COMPLETE":
            self.complete_checkpoint_count += 1
        elif status == "BID_ONLY":
            self.bid_only_checkpoint_count += 1
        elif status == "ASK_ONLY":
            self.ask_only_checkpoint_count += 1
        elif status == "EMPTY_BOTH":
            self.empty_checkpoint_count += 1
        else:
            self.invalid_checkpoint_count += 1

        # Once a diff chain is active (or a candidate is already awaiting its
        # handshake), later periodic snapshots are observations only.
        if self.valid or self.bridge_pending:
            self.audit_only_checkpoint_count += 1
            if status == "COMPLETE" and self.valid:
                bid, bid_q = self.bids.best()
                ask, ask_q = self.asks.best()
                if snap_bid_top == (bid, bid_q) and snap_ask_top == (ask, ask_q):
                    self.complete_checkpoint_match_count += 1
                else:
                    self.complete_checkpoint_mismatch_count += 1
            return

        if status != "COMPLETE" or last_u_i is None:
            return

        self.bootstrap_candidate_count += 1
        snap_bids = CompiledBookSide(True, self.price_scale)
        snap_asks = CompiledBookSide(False, self.price_scale)
        snap_bids.reset(row.get("bids") or [])
        snap_asks.reset(row.get("asks") or [])
        self.bids = snap_bids
        self.asks = snap_asks
        self.snapshot_last_update_id = last_u_i
        self.previous_u = None
        self.bridge_pending = True
        self.valid = False
        self.last_bridge_mode = "LOCAL_CHECKPOINT_PENDING"
        # Visible checkpoint state exists at ts, but execution rows remain
        # invalid until the first following diff proves exact local continuity.
        self.last_state_ts_ns = ts
        self.last_bootstrap_ts_ns = ts

    def _record_actions(self, is_bid: bool, count: int, actions: tuple[int, int, int, int]) -> None:
        if is_bid:
            self.bid_update_count += int(count)
        else:
            self.ask_update_count += int(count)
        ins, mod, dele, noop = actions
        self.price_insert_count += int(ins)
        self.price_modify_count += int(mod)
        self.price_delete_count += int(dele)
        self.price_noop_count += int(noop)

    def _apply_row_levels(self, row: Mapping[str, Any]) -> None:
        """Apply one depth message using vectorized tick arrays when available."""
        if "_fast_bid_ticks" in row:
            bt = row.get("_fast_bid_ticks")
            bq = row.get("_fast_bid_qty")
            at = row.get("_fast_ask_ticks")
            aq = row.get("_fast_ask_qty")
            bt = bt if bt is not None else _np.empty(0, dtype=_np.int64)
            bq = bq if bq is not None else _np.empty(0, dtype=_np.float64)
            at = at if at is not None else _np.empty(0, dtype=_np.int64)
            aq = aq if aq is not None else _np.empty(0, dtype=_np.float64)
            self._record_actions(True, len(bt), self.bids.update_arrays(bt, bq))
            self._record_actions(False, len(at), self.asks.update_arrays(at, aq))
        else:
            for side, levels, is_bid in (
                (self.bids, row.get("bid_updates") or [], True),
                (self.asks, row.get("ask_updates") or [], False),
            ):
                ticks: list[int] = []
                qtys: list[float] = []
                for level in levels:
                    if isinstance(level, Mapping):
                        p = safe_float(level.get("price"))
                        q = safe_float(level.get("quantity"))
                    else:
                        try:
                            p = safe_float(level[0])
                            q = safe_float(level[1])
                        except Exception:
                            continue
                    if p is None or q is None:
                        continue
                    ticks.append(side._price_to_tick(p))
                    qtys.append(q)
                arr_t = _np.asarray(ticks, dtype=_np.int64)
                arr_q = _np.asarray(qtys, dtype=_np.float64)
                self._record_actions(is_bid, len(arr_t), side.update_arrays(arr_t, arr_q))
        self.max_bid_levels = max(self.max_bid_levels, len(self.bids))
        self.max_ask_levels = max(self.max_ask_levels, len(self.asks))

    def apply_depth(self, row: Mapping[str, Any], gap_records: list[GapRecord], day_text: str) -> None:
        ts = int(row["ts_recv_ns"])
        self.depth_count += 1
        U = int(row["first_update_id"])
        u = int(row["final_update_id"])
        pu = int(row["previous_final_update_id"])

        if self.bridge_pending:
            L = int(self.snapshot_last_update_id) if self.snapshot_last_update_id is not None else None
            if L is None:
                self.bridge_pending = False
                self.ignored_invalid_depth_count += 1
                return
            if u <= L:
                self.stale_depth_count += 1
                return

            # Stored snapshots are collector-local checkpoints.  The only valid
            # historical bootstrap handshake is exact continuation from last_u.
            if pu != L:
                self.bootstrap_bridge_failure_count += 1
                gap_records.append(
                    GapRecord(day_text, ts, "BOOTSTRAP_LOCAL_BRIDGE_FAILED", None, pu, U, u, L)
                )
                self.invalidate(ts, "BOOTSTRAP_LOCAL_BRIDGE_FAILED")
                return

            self._apply_row_levels(row)
            self.previous_u = u
            self.bridge_pending = False
            self.last_bridge_mode = "LOCAL_CHECKPOINT_CONTINUATION"
            self.last_state_ts_ns = ts
            if not self._top_ok():
                gap_records.append(GapRecord(day_text, ts, "CROSSED_BOOK", self.previous_u, pu, U, u, L))
                self.invalidate(ts, "CROSSED_BOOK")
                return

            self.valid = True
            self.bootstrap_count += 1
            if self.first_bootstrap_ts_ns is None:
                self.first_bootstrap_ts_ns = self.last_bootstrap_ts_ns
            else:
                self.recovery_count += 1
            self.bootstrap_source = "COMPLETE_LOCALBOOK_CHECKPOINT"
            if self.invalid_since_ns is not None and self.last_bootstrap_ts_ns is not None:
                self.recovery_latencies_ms.append((ts - self.invalid_since_ns) / 1_000_000.0)
                self.invalid_since_ns = None
            return

        if not self.valid:
            self.ignored_invalid_depth_count += 1
            return

        if self.previous_u is None or pu != self.previous_u:
            gap_records.append(
                GapRecord(day_text, ts, "PU_SEQUENCE_GAP", self.previous_u, pu, U, u, self.snapshot_last_update_id)
            )
            self.invalidate(ts, "PU_SEQUENCE_GAP")
            return

        self._apply_row_levels(row)
        self.previous_u = u
        self.last_state_ts_ns = ts
        if not self._top_ok():
            gap_records.append(
                GapRecord(day_text, ts, "CROSSED_BOOK", self.previous_u, pu, U, u, self.snapshot_last_update_id)
            )
            self.invalidate(ts, "CROSSED_BOOK")

    def metrics(self, sample_second: int, notionals: Sequence[float], fee_bps: float, max_age_ms: int, symbol: str = DEFAULT_SYMBOL) -> dict[str, Any]:
        sample_end_ns = (sample_second + 1) * 1_000_000_000 - 1
        age_ms = (
            (sample_end_ns - self.last_state_ts_ns) / 1_000_000.0
            if self.last_state_ts_ns is not None
            else math.nan
        )
        age_valid = bool(math.isfinite(age_ms) and age_ms <= max_age_ms)
        usable = bool(self.valid and not self.bridge_pending and age_valid and self._top_ok())
        bid, bid_q = self.bids.best()
        ask, ask_q = self.asks.best()
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else math.nan
        spread_bps = ((ask - bid) / mid * 10_000.0) if usable and mid > 0 else math.nan
        row: dict[str, Any] = {
            "execution_schema_version": OUTPUT_SCHEMA_VERSION,
            "symbol": symbol,
            "receive_second": int(sample_second),
            "sample_time": datetime.fromtimestamp(sample_second, tz=timezone.utc),
            "book_valid": usable,
            "sequence_valid": bool(self.valid and not self.bridge_pending),
            "bridge_pending": bool(self.bridge_pending),
            "bridge_mode": self.last_bridge_mode,
            "bootstrap_source": self.bootstrap_source,
            "first_bootstrap_ts_recv_ns": self.first_bootstrap_ts_ns,
            "last_bootstrap_ts_recv_ns": self.last_bootstrap_ts_ns,
            "state_age_ms": float(age_ms) if math.isfinite(age_ms) else math.nan,
            "snapshot_age_ms": (
                (sample_end_ns - self.last_snapshot_ts_ns) / 1_000_000.0
                if self.last_snapshot_ts_ns is not None
                else math.nan
            ),
            "last_state_ts_recv_ns": self.last_state_ts_ns,
            "last_snapshot_ts_recv_ns": self.last_snapshot_ts_ns,
            "last_update_id": self.previous_u if self.previous_u is not None else self.snapshot_last_update_id,
            "best_bid": float(bid) if usable and bid is not None else math.nan,
            "best_ask": float(ask) if usable and ask is not None else math.nan,
            "best_bid_qty": float(bid_q) if usable and bid_q is not None else math.nan,
            "best_ask_qty": float(ask_q) if usable and ask_q is not None else math.nan,
            "mid_price": float(mid) if usable else math.nan,
            "spread_bps": float(spread_bps) if usable else math.nan,
            "taker_fee_assumption_bps": float(fee_bps),
        }
        if usable:
            bid_depths, sell_exec = self.bids.scan_depth_and_execution(mid, DEPTH_BANDS_BPS, notionals)
            ask_depths, buy_exec = self.asks.scan_depth_and_execution(mid, DEPTH_BANDS_BPS, notionals)
        else:
            bid_depths = ask_depths = {}
            buy_exec = sell_exec = {}

        for band in DEPTH_BANDS_BPS:
            tag = str(int(band))
            row[f"bid_depth_quote_{tag}bps"] = bid_depths.get(float(band), math.nan) if usable else math.nan
            row[f"ask_depth_quote_{tag}bps"] = ask_depths.get(float(band), math.nan) if usable else math.nan
        for n in notionals:
            tag = str(int(round(n)))
            if not usable:
                for prefix in ("buy", "sell"):
                    row[f"{prefix}_vwap_{tag}"] = math.nan
                    row[f"{prefix}_filled_quote_{tag}"] = 0.0
                    row[f"{prefix}_fill_fraction_{tag}"] = 0.0
                    row[f"{prefix}_complete_{tag}"] = False
                    row[f"{prefix}_crossing_cost_bps_{tag}"] = math.nan
                    row[f"{prefix}_total_cost_bps_{tag}"] = math.nan
                continue
            bvwap, bq, _, bcomplete = buy_exec[float(n)]
            svwap, sq, _, scomplete = sell_exec[float(n)]
            buy_cost = bps_from_ratio(bvwap, mid) if bcomplete else math.nan
            sell_cost = -bps_from_ratio(svwap, mid) if scomplete else math.nan
            row[f"buy_vwap_{tag}"] = bvwap if bcomplete else math.nan
            row[f"buy_filled_quote_{tag}"] = bq
            row[f"buy_fill_fraction_{tag}"] = min(1.0, bq / n) if n > 0 else math.nan
            row[f"buy_complete_{tag}"] = bool(bcomplete)
            row[f"buy_crossing_cost_bps_{tag}"] = buy_cost
            row[f"buy_total_cost_bps_{tag}"] = buy_cost + fee_bps if bcomplete else math.nan
            row[f"sell_vwap_{tag}"] = svwap if scomplete else math.nan
            row[f"sell_filled_quote_{tag}"] = sq
            row[f"sell_fill_fraction_{tag}"] = min(1.0, sq / n) if n > 0 else math.nan
            row[f"sell_complete_{tag}"] = bool(scomplete)
            row[f"sell_crossing_cost_bps_{tag}"] = sell_cost
            row[f"sell_total_cost_bps_{tag}"] = sell_cost + fee_bps if scomplete else math.nan
        return row


def discover_family_days(root: Path, family: str) -> dict[date, tuple[Path, ...]]:
    family_root = root / "perps" / family
    out: dict[date, tuple[Path, ...]] = {}
    if not family_root.exists():
        return out
    for ddir in sorted(family_root.glob("date=????-??-??")):
        if not ddir.is_dir():
            continue
        try:
            d = date.fromisoformat(ddir.name.split("=", 1)[1])
        except Exception:
            continue
        files = tuple(sorted(x for x in ddir.glob("*.parquet") if x.is_file()))
        if files:
            out[d] = files
    return out


def discover_days(canonical_root: Path, start: date | None, end: date | None, max_dates: int | None) -> list[DayFiles]:
    fam = {
        name: discover_family_days(canonical_root, name)
        for name in ("depth_updates", "book_snapshots", "trades", "liquidations")
    }
    days = sorted(set(fam["depth_updates"]) & set(fam["book_snapshots"]))
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    if max_dates is not None:
        days = days[:max_dates]
    return [
        DayFiles(
            day=d,
            depth=fam["depth_updates"].get(d, ()),
            snapshots=fam["book_snapshots"].get(d, ()),
            trades=fam["trades"].get(d, ()),
            liquidations=fam["liquidations"].get(d, ()),
        )
        for d in days
    ]


def iter_parquet_rows(
    paths: Sequence[Path],
    columns: Sequence[str],
    symbol: str | None = None,
    batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Stream selected parquet columns without materializing list columns in Pandas.

    Canonical depth rows contain nested bid/ask update arrays. Converting a full
    day to Pandas creates a multi-million-element object ndarray and can exhaust
    memory on normal research machines. Arrow batches keep peak memory bounded.
    """
    _, _, _, pq = require_data_stack()
    wanted_symbol = symbol.upper() if symbol else None
    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            available = set(pf.schema_arrow.names)
            use = [c for c in columns if c in available]
            last_key: tuple[int, int] | None = None
            for batch in pf.iter_batches(batch_size=batch_size, columns=use):
                # Batch-sized Python objects are intentional here: LocalBook needs
                # the nested list values, but the entire day must never coexist.
                rows = batch.to_pylist()
                for row in rows:
                    if wanted_symbol and str(row.get("symbol", "")).upper() != wanted_symbol:
                        continue
                    ts = row.get("ts_recv_ns")
                    if ts is None:
                        continue
                    try:
                        ts_i = int(ts)
                    except (TypeError, ValueError):
                        continue
                    sr = row.get("source_record_number")
                    try:
                        sr_i = int(sr) if sr is not None else 0
                    except (TypeError, ValueError):
                        sr_i = 0
                    key = (ts_i, sr_i)
                    if last_key is not None and key < last_key:
                        raise RuntimeError(f"Canonical parquet is not receive-time sorted: {path}")
                    last_key = key
                    row["ts_recv_ns"] = ts_i
                    row["source_record_number"] = sr_i
                    yield row
        finally:
            close = getattr(pf, "close", None)
            if callable(close):
                close()


def _arrow_list_struct_numeric(array, price_scale: float, path: Path, column_name: str):
    """Flatten canonical list<struct<price:string,quantity:string>> in Arrow/C++.

    This avoids materializing tens of millions of Python dicts and calling
    float() once per price level.  Tick alignment is checked before rounding so
    the compiled book cannot silently alter exchange prices.
    """
    np, _, pa, _ = require_data_stack()
    import pyarrow.compute as pc  # type: ignore
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise RuntimeError(f"{path}: {column_name} is not a canonical list array: {array.type}")
    offsets_abs = array.offsets.to_numpy(zero_copy_only=False).astype("int64", copy=False)
    if len(offsets_abs) == 0:
        return np.asarray([0], dtype="int64"), np.empty(0, dtype="int64"), np.empty(0, dtype="float64")
    base = int(offsets_abs[0])
    stop = int(offsets_abs[-1])
    offsets = offsets_abs - base
    values = array.values.slice(base, max(0, stop - base))
    if not pa.types.is_struct(values.type):
        raise RuntimeError(f"{path}: {column_name} values are not canonical structs: {values.type}")
    try:
        price_col = values.field("price")
        qty_col = values.field("quantity")
    except Exception as exc:
        raise RuntimeError(f"{path}: {column_name} lacks price/quantity struct fields") from exc
    prices = pc.cast(price_col, pa.float64(), safe=False).to_numpy(zero_copy_only=False)
    qtys = pc.cast(qty_col, pa.float64(), safe=False).to_numpy(zero_copy_only=False)
    prices = np.asarray(prices, dtype="float64")
    qtys = np.asarray(qtys, dtype="float64")
    finite = np.isfinite(prices) & (prices > 0)
    scaled = prices[finite] * float(price_scale)
    if len(scaled):
        rounded = np.rint(scaled)
        err = np.abs(scaled - rounded)
        if np.any(err > 1e-6):
            bad_i = int(np.flatnonzero(err > 1e-6)[0])
            bad_price = float(prices[finite][bad_i])
            raise RuntimeError(
                f"{path}: price {bad_price} in {column_name} is not aligned to --price-scale {price_scale:g}. "
                "Use a finer scale, e.g. --price-scale 100."
            )
    ticks = np.zeros(len(prices), dtype="int64")
    ticks[finite] = np.rint(prices[finite] * float(price_scale)).astype("int64")
    return offsets.astype("int64", copy=False), ticks, qtys


def iter_fast_depth_rows(
    paths: Sequence[Path],
    columns: Sequence[str],
    symbol: str,
    price_scale: float,
    stats: FastDepthStats,
    batch_size: int = DEPTH_STREAM_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Stream depth metadata row-wise but nested levels as numeric NumPy views."""
    np, _, pa, pq = require_data_stack()
    import pyarrow.compute as pc  # type: ignore
    wanted_symbol = symbol.upper()
    for path in paths:
        pf = pq.ParquetFile(path)
        last_key: tuple[int, int] | None = None
        try:
            available = set(pf.schema_arrow.names)
            use = [c for c in columns if c in available]
            for batch in pf.iter_batches(batch_size=batch_size, columns=use):
                t0 = time.perf_counter()
                if "symbol" in batch.schema.names:
                    sym = batch.column(batch.schema.get_field_index("symbol"))
                    mask = pc.equal(sym, pa.scalar(wanted_symbol))
                    # Filtering in Arrow preserves nested-list offsets and avoids
                    # Python symbol checks for every depth message.
                    if not bool(pc.all(pc.fill_null(mask, False)).as_py()):
                        batch = batch.filter(pc.fill_null(mask, False))
                if batch.num_rows == 0:
                    stats.arrow_decode_seconds += time.perf_counter() - t0
                    continue

                def int_col(name: str, default: int = 0):
                    if name not in batch.schema.names:
                        return np.full(batch.num_rows, default, dtype="int64")
                    arr = batch.column(batch.schema.get_field_index(name))
                    arr = pc.cast(pc.fill_null(arr, default), pa.int64(), safe=False)
                    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype="int64")

                ts = int_col("ts_recv_ns")
                U = int_col("first_update_id")
                u = int_col("final_update_id")
                pu = int_col("previous_final_update_id")
                sr = int_col("source_record_number")
                b_arr = batch.column(batch.schema.get_field_index("bid_updates"))
                a_arr = batch.column(batch.schema.get_field_index("ask_updates"))
                boff, bticks, bqty = _arrow_list_struct_numeric(b_arr, price_scale, path, "bid_updates")
                aoff, aticks, aqty = _arrow_list_struct_numeric(a_arr, price_scale, path, "ask_updates")

                if len(ts) > 1:
                    bad = (ts[1:] < ts[:-1]) | ((ts[1:] == ts[:-1]) & (sr[1:] < sr[:-1]))
                    if bool(np.any(bad)):
                        raise RuntimeError(f"Canonical parquet is not receive-time sorted: {path}")
                if len(ts):
                    first_key = (int(ts[0]), int(sr[0]))
                    if last_key is not None and first_key < last_key:
                        raise RuntimeError(f"Canonical parquet is not receive-time sorted: {path}")
                    last_key = (int(ts[-1]), int(sr[-1]))

                stats.arrow_batches += 1
                stats.rows_decoded += int(batch.num_rows)
                stats.levels_decoded += int(len(bticks) + len(aticks))
                stats.arrow_decode_seconds += time.perf_counter() - t0

                for i in range(batch.num_rows):
                    b0, b1 = int(boff[i]), int(boff[i + 1])
                    a0, a1 = int(aoff[i]), int(aoff[i + 1])
                    yield {
                        "ts_recv_ns": int(ts[i]),
                        "first_update_id": int(U[i]),
                        "final_update_id": int(u[i]),
                        "previous_final_update_id": int(pu[i]),
                        "source_record_number": int(sr[i]),
                        "_fast_bid_ticks": bticks[b0:b1],
                        "_fast_bid_qty": bqty[b0:b1],
                        "_fast_ask_ticks": aticks[a0:a1],
                        "_fast_ask_qty": aqty[a0:a1],
                    }
        finally:
            close = getattr(pf, "close", None)
            if callable(close):
                close()


def merge_sorted_row_iters(iters: Sequence[Iterator[dict[str, Any]]], priority: int, kind: str) -> Iterator[dict[str, Any]]:
    """K-way merge already receive-time-sorted canonical shards."""
    heap: list[tuple[int, int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for idx, it in enumerate(iters):
        try:
            row = next(it)
        except StopIteration:
            continue
        heap.append((int(row["ts_recv_ns"]), int(row.get("source_record_number", 0)), idx, row, it))
    heapq.heapify(heap)
    while heap:
        _, _, idx, row, it = heapq.heappop(heap)
        row["_kind"] = kind
        row["_priority"] = priority
        yield row
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(nxt["ts_recv_ns"]), int(nxt.get("source_record_number", 0)), idx, nxt, it))


def read_parquets(paths: Sequence[Path], columns: Sequence[str] | None = None):
    """Small/scalar-family helper; nested depth data must use iter_parquet_rows."""
    _, pd, _, pq = require_data_stack()
    frames = []
    for p in paths:
        pf = pq.ParquetFile(p)
        try:
            available = set(pf.schema_arrow.names)
            use = [c for c in columns if c in available] if columns else None
            for batch in pf.iter_batches(batch_size=65536, columns=use):
                frames.append(batch.to_pandas())
        finally:
            close = getattr(pf, "close", None)
            if callable(close):
                close()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def filter_symbol(frame, symbol: str):
    if frame.empty or "symbol" not in frame.columns:
        return frame
    return frame.loc[frame["symbol"].astype(str).str.upper() == symbol.upper()].copy()


def validate_family_schema(day_files: DayFiles) -> list[Check]:
    _, _, _, pq = require_data_stack()
    checks: list[Check] = []
    required = {
        "depth": (day_files.depth, {"ts_recv_ns", "symbol", "first_update_id", "final_update_id", "previous_final_update_id", "bid_updates", "ask_updates"}),
        "snapshots": (day_files.snapshots, {"ts_recv_ns", "symbol", "last_update_id", "bids", "asks"}),
        "trades": (day_files.trades, {"ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker"}),
        "liquidations": (day_files.liquidations, {"ts_recv_ns", "symbol", "exchange_event_ms", "side", "original_quantity", "price", "average_price", "last_filled_quantity"}),
    }
    for family, (paths, cols) in required.items():
        if not paths:
            status = "FAIL" if family in {"depth", "snapshots"} else "WARN"
            checks.append(Check("schema", f"{family}/files_present", status, 0, ">=1", "Optional" if status == "WARN" else "Required"))
            continue
        missing_union: Counter[str] = Counter()
        for p in paths:
            names = set(pq.ParquetFile(p).schema_arrow.names)
            for col in cols - names:
                missing_union[col] += 1
        checks.append(
            Check(
                "schema",
                f"{family}/required_columns",
                "PASS" if not missing_union else "FAIL",
                dict(missing_union),
                {},
            )
        )
    return checks


def unified_book_events(day_files: DayFiles, symbol: str, price_scale: float, fast_stats: FastDepthStats) -> Iterator[dict[str, Any]]:
    """Stream snapshots and diff-depth updates in deterministic receive-time order."""
    depth_cols = [
        "ts_recv_ns", "symbol", "first_update_id", "final_update_id",
        "previous_final_update_id", "bid_updates", "ask_updates",
        "source_file_id", "source_record_number",
    ]
    snap_cols = [
        "ts_recv_ns", "symbol", "last_update_id", "bids", "asks",
        "source_file_id", "source_record_number",
    ]
    depth_iters = [
        iter_fast_depth_rows((p,), depth_cols, symbol=symbol, price_scale=price_scale, stats=fast_stats, batch_size=DEPTH_STREAM_BATCH_SIZE)
        for p in day_files.depth
    ]
    # Full-book snapshots carry much larger nested arrays than diff updates.
    snap_iters = [iter_parquet_rows((p,), snap_cols, symbol=symbol, batch_size=SNAPSHOT_STREAM_BATCH_SIZE) for p in day_files.snapshots]
    depth = merge_sorted_row_iters(depth_iters, priority=1, kind="DEPTH")
    snaps = merge_sorted_row_iters(snap_iters, priority=0, kind="SNAPSHOT")

    # Merge family streams. Snapshot wins an exact receive-time/source-record tie.
    heap: list[tuple[int, int, int, int, dict[str, Any], Iterator[dict[str, Any]]]] = []
    for family_idx, it in enumerate((snaps, depth)):
        try:
            row = next(it)
        except StopIteration:
            continue
        heap.append((int(row["ts_recv_ns"]), int(row["_priority"]), int(row.get("source_record_number", 0)), family_idx, row, it))
    heapq.heapify(heap)
    while heap:
        _, _, _, family_idx, row, it = heapq.heappop(heap)
        yield row
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (int(nxt["ts_recv_ns"]), int(nxt["_priority"]), int(nxt.get("source_record_number", 0)), family_idx, nxt, it))

def aggregate_trades(day_files: DayFiles, symbol: str):
    """Aggregate scalar trade rows to seconds in bounded Arrow/Pandas batches."""
    np, pd, _, pq = require_data_stack()
    cols = ["ts_recv_ns", "symbol", "price", "quantity", "buyer_is_maker"]
    parts = []
    for path in day_files.trades:
        pf = pq.ParquetFile(path)
        try:
            available = set(pf.schema_arrow.names)
            use = [c for c in cols if c in available]
            for batch in pf.iter_batches(batch_size=131072, columns=use):
                f = batch.to_pandas()
                f = filter_symbol(f, symbol)
                if f.empty:
                    continue
                f["ts_recv_ns"] = pd.to_numeric(f["ts_recv_ns"], errors="coerce")
                f["price"] = pd.to_numeric(f["price"], errors="coerce")
                f["quantity"] = pd.to_numeric(f["quantity"], errors="coerce")
                f = f.dropna(subset=["ts_recv_ns", "price", "quantity"])
                f = f[(f["price"] > 0) & (f["quantity"] >= 0)].copy()
                if f.empty:
                    continue
                f["receive_second"] = (f["ts_recv_ns"].astype("int64") // 1_000_000_000).astype("int64")
                f["quote"] = f["price"] * f["quantity"]
                maker = f["buyer_is_maker"].fillna(False).astype(bool)
                f["signed"] = np.where(maker, -f["quote"], f["quote"])
                f = f.sort_values("ts_recv_ns")
                g = f.groupby("receive_second", sort=True)
                part = g.agg(
                    trade_count=("quote", "size"),
                    trade_quote_volume=("quote", "sum"),
                    signed_aggressive_quote=("signed", "sum"),
                    last_trade_price=("price", "last"),
                    last_trade_ts=("ts_recv_ns", "last"),
                ).reset_index()
                parts.append(part)
        finally:
            close = getattr(pf, "close", None)
            if callable(close):
                close()
    if not parts:
        return pd.DataFrame(columns=["receive_second", "trade_count", "trade_quote_volume", "signed_aggressive_quote", "last_trade_price"])
    allp = pd.concat(parts, ignore_index=True)
    # Sum additive fields across batches/shards; select the chronologically last price.
    sums = allp.groupby("receive_second", sort=True).agg(
        trade_count=("trade_count", "sum"),
        trade_quote_volume=("trade_quote_volume", "sum"),
        signed_aggressive_quote=("signed_aggressive_quote", "sum"),
    )
    last = allp.sort_values(["receive_second", "last_trade_ts"]).groupby("receive_second", sort=True).tail(1).set_index("receive_second")[["last_trade_price"]]
    return sums.join(last).reset_index()

def aggregate_liquidations(day_files: DayFiles, symbol: str):
    np, pd, _, _ = require_data_stack()
    cols = [
        "ts_recv_ns", "symbol", "exchange_event_ms", "trade_time_ms", "side",
        "original_quantity", "price", "average_price", "last_filled_quantity",
        "cumulative_filled_quantity",
    ]
    f = filter_symbol(read_parquets(day_files.liquidations, cols), symbol)
    if f.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["receive_second", "force_order_count", "force_sell_count", "force_buy_count", "force_selected_notional_sum"])
    for c in ("ts_recv_ns", "exchange_event_ms", "trade_time_ms", "original_quantity", "price", "average_price", "last_filled_quantity", "cumulative_filled_quantity"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.dropna(subset=["ts_recv_ns"]).copy()
    f["receive_second"] = (f["ts_recv_ns"].astype("int64") // 1_000_000_000).astype("int64")
    f["exchange_second"] = (f["exchange_event_ms"] // 1000).astype("Int64") if "exchange_event_ms" in f.columns else pd.Series([pd.NA] * len(f), dtype="Int64")
    f["exchange_event_position_ms"] = (f["exchange_event_ms"] % 1000).astype("float") if "exchange_event_ms" in f.columns else math.nan
    f["trade_time_position_ms"] = (f["trade_time_ms"] % 1000).astype("float") if "trade_time_ms" in f.columns else math.nan
    # Use the most local filled-size field available. This remains a censored
    # selected observation and is never interpreted as total liquidation volume.
    qty = f.get("last_filled_quantity", pd.Series(index=f.index, dtype=float)).fillna(0.0)
    if "cumulative_filled_quantity" in f:
        qty = qty.where(qty > 0, f["cumulative_filled_quantity"].fillna(0.0))
    if "original_quantity" in f:
        qty = qty.where(qty > 0, f["original_quantity"].fillna(0.0))
    px = f.get("average_price", pd.Series(index=f.index, dtype=float)).fillna(0.0)
    if "price" in f:
        px = px.where(px > 0, f["price"].fillna(0.0))
    f["observed_selected_notional"] = qty * px
    side = f.get("side", pd.Series(index=f.index, dtype=str)).astype(str).str.upper()
    f["force_sell"] = (side == "SELL").astype(int)
    f["force_buy"] = (side == "BUY").astype(int)
    g = f.groupby("receive_second", sort=True)
    agg = g.agg(
        force_order_count=("observed_selected_notional", "size"),
        force_sell_count=("force_sell", "sum"),
        force_buy_count=("force_buy", "sum"),
        force_selected_notional_sum=("observed_selected_notional", "sum"),
    ).reset_index()
    return f, agg


def output_path(output_root: Path, symbol: str, d: date) -> Path:
    return output_root / f"symbol={symbol}" / f"date={d.isoformat()}" / f"part-{d.isoformat()}.parquet"


def quality_sidecar_path(output_root: Path, symbol: str, d: date) -> Path:
    return output_root / "quality" / f"symbol={symbol}" / f"date={d.isoformat()}.json"


def atomic_write_parquet(path: Path, frame, metadata: Mapping[str, Any]) -> None:
    _, _, pa, pq = require_data_stack()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = short_tmp(path, "pq")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    md = {k.encode(): str(v).encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **md})
    try:
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True, write_statistics=True)
        check = pq.ParquetFile(tmp)
        try:
            if int(check.metadata.num_rows) != len(frame):
                raise RuntimeError("Atomic parquet row-count validation failed")
        finally:
            close = getattr(check, "close", None)
            if callable(close):
                close()
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def init_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS day_builds(
            receive_date TEXT PRIMARY KEY,
            input_signature TEXT NOT NULL,
            builder_version TEXT NOT NULL,
            status TEXT NOT NULL,
            output_rows INTEGER NOT NULL DEFAULT 0,
            output_path TEXT,
            run_id TEXT,
            completed_at TEXT,
            error TEXT
        )
        """
    )
    conn.commit()
    return conn


def can_reuse(conn: sqlite3.Connection, d: date, sig: str, out: Path, sidecar: Path) -> bool:
    row = conn.execute(
        "SELECT input_signature,builder_version,status,output_rows,output_path FROM day_builds WHERE receive_date=?",
        (d.isoformat(),),
    ).fetchone()
    if not row or row[0] != sig or row[1] != BUILDER_VERSION or row[2] != "COMPLETED" or not out.exists() or not sidecar.exists():
        return False
    try:
        _, _, _, pq = require_data_stack()
        return int(pq.ParquetFile(out).metadata.num_rows) == int(row[3])
    except Exception:
        return False


def mark_state(conn: sqlite3.Connection, d: date, sig: str, status: str, rows: int, out: Path, run_id: str, error: str = "") -> None:
    conn.execute(
        """
        INSERT INTO day_builds(receive_date,input_signature,builder_version,status,output_rows,output_path,run_id,completed_at,error)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(receive_date) DO UPDATE SET
          input_signature=excluded.input_signature,
          builder_version=excluded.builder_version,
          status=excluded.status,
          output_rows=excluded.output_rows,
          output_path=excluded.output_path,
          run_id=excluded.run_id,
          completed_at=excluded.completed_at,
          error=excluded.error
        """,
        (d.isoformat(), sig, BUILDER_VERSION, status, int(rows), str(out), run_id, utc_now_iso(), error),
    )
    conn.commit()


def build_day(
    day_files: DayFiles,
    symbol: str,
    notionals: Sequence[float],
    fee_bps: float,
    max_age_ms: int,
    progress_every_depth: int = DEFAULT_PROGRESS_EVERY_DEPTH,
    price_scale: float = DEFAULT_PRICE_SCALE,
):
    np, pd, _, _ = require_data_stack()
    day_text = day_files.day.isoformat()
    day_started = time.perf_counter()
    stage_started = time.perf_counter()
    checks = validate_family_schema(day_files)
    schema_seconds = time.perf_counter() - stage_started
    if any(c.status == "FAIL" for c in checks):
        raise RuntimeError(f"Required canonical schema check failed for {day_text}")

    fast_stats = FastDepthStats()
    events = unified_book_events(day_files, symbol, price_scale, fast_stats)
    stage_started = time.perf_counter()
    trades = aggregate_trades(day_files, symbol)
    trade_aggregation_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    liq_raw, liq_agg = aggregate_liquidations(day_files, symbol)
    liquidation_aggregation_seconds = time.perf_counter() - stage_started
    start_sec = int(datetime(day_files.day.year, day_files.day.month, day_files.day.day, tzinfo=timezone.utc).timestamp())
    end_sec = start_sec + SECONDS_PER_DAY
    book = LocalBook(price_scale)
    gap_records: list[GapRecord] = []
    rows: list[dict[str, Any]] = []
    current_sec = start_sec
    cached_metrics: dict[str, Any] | None = None
    dirty = True
    compiled_metric_seconds = 0.0

    def timed_metrics(sec: int, ns: Sequence[float]):
        nonlocal compiled_metric_seconds
        mt0 = time.perf_counter()
        out = book.metrics(sec, ns, fee_bps, max_age_ms, symbol)
        compiled_metric_seconds += time.perf_counter() - mt0
        return out

    def flush_until(target_sec: int) -> None:
        nonlocal current_sec, cached_metrics, dirty
        while current_sec < target_sec and current_sec < end_sec:
            if dirty or cached_metrics is None:
                cached_metrics = timed_metrics(current_sec, notionals)
                dirty = False
            else:
                # Recompute only age/validity-sensitive fields while preserving book economics.
                base = dict(cached_metrics)
                fresh = timed_metrics(current_sec, ())
                for k in (
                    "receive_second", "sample_time", "book_valid", "sequence_valid", "bridge_pending", "bridge_mode",
                    "bootstrap_source", "first_bootstrap_ts_recv_ns", "last_bootstrap_ts_recv_ns",
                    "state_age_ms", "snapshot_age_ms", "last_state_ts_recv_ns", "last_snapshot_ts_recv_ns",
                    "last_update_id", "best_bid", "best_ask", "best_bid_qty", "best_ask_qty", "mid_price", "spread_bps",
                ):
                    if k in fresh:
                        base[k] = fresh[k]
                # If freshness invalidated the book, economic columns must be cleared too.
                if not base["book_valid"]:
                    base = timed_metrics(current_sec, notionals)
                cached_metrics = base
            rows.append(dict(cached_metrics))
            current_sec += 1

    event_count = 0
    replay_started = time.perf_counter()
    next_progress_depth = int(progress_every_depth) if progress_every_depth and progress_every_depth > 0 else None
    for rec in events:
        ts = int(rec["ts_recv_ns"])
        sec = ts // 1_000_000_000
        if sec < start_sec or sec >= end_sec:
            continue
        event_count += 1
        flush_until(sec)
        if rec.get("_kind") == "SNAPSHOT":
            book.apply_snapshot(rec)
        else:
            book.apply_depth(rec, gap_records, day_text)
            if next_progress_depth is not None and book.depth_count >= next_progress_depth:
                elapsed = max(1e-9, time.perf_counter() - replay_started)
                rate = book.depth_count / elapsed
                print(
                    f"    {day_text}: depth={book.depth_count:,}, "
                    f"rate={rate:,.0f}/s, valid={'Y' if book.valid else 'N'}, "
                    f"levels={len(book.bids):,}/{len(book.asks):,}"
                )
                next_progress_depth += int(progress_every_depth)
        dirty = True
    if event_count == 0:
        raise RuntimeError(f"No canonical order-book events for {day_text}")
    flush_until(end_sec)
    replay_sampling_seconds = time.perf_counter() - replay_started

    enrichment_started = time.perf_counter()
    frame = pd.DataFrame(rows)
    # Enrich with contemporaneous flow/liquidation data without forward fill.
    if not trades.empty:
        frame = frame.merge(trades, on="receive_second", how="left")
    else:
        frame["trade_count"] = 0
        frame["trade_quote_volume"] = 0.0
        frame["signed_aggressive_quote"] = 0.0
        frame["last_trade_price"] = math.nan
    if not liq_agg.empty:
        frame = frame.merge(liq_agg, on="receive_second", how="left")
    else:
        for c in ("force_order_count", "force_sell_count", "force_buy_count", "force_selected_notional_sum"):
            frame[c] = 0.0
    for c in ("trade_count", "trade_quote_volume", "signed_aggressive_quote", "force_order_count", "force_sell_count", "force_buy_count", "force_selected_notional_sum"):
        if c in frame:
            frame[c] = frame[c].fillna(0)

    # Causal last-price series for descriptive data-quality stress stratification.
    frame["trade_price_ffill"] = frame["last_trade_price"].ffill()
    frame["abs_return_1s_bps"] = (frame["trade_price_ffill"].pct_change().abs() * 10_000.0).replace([np.inf, -np.inf], np.nan)
    frame["aggressive_imbalance"] = np.where(
        frame["trade_quote_volume"] > 0,
        frame["signed_aggressive_quote"] / frame["trade_quote_volume"],
        0.0,
    )
    frame["force_order_present"] = frame["force_order_count"] > 0
    enrichment_seconds = time.perf_counter() - enrichment_started

    # Daily quality summary.
    valid = frame["book_valid"].astype(bool)
    daily = {
        "receive_date": day_text,
        "rows": len(frame),
        "book_valid_seconds": int(valid.sum()),
        "book_valid_rate": float(valid.mean()),
        "sequence_gap_count": int(book.actual_sequence_gap_count),
        "crossed_book_count": int(book.crossed_count),
        "snapshot_count": int(book.snapshot_count),
        "complete_checkpoint_count": int(book.complete_checkpoint_count),
        "bid_only_checkpoint_count": int(book.bid_only_checkpoint_count),
        "ask_only_checkpoint_count": int(book.ask_only_checkpoint_count),
        "empty_checkpoint_count": int(book.empty_checkpoint_count),
        "invalid_checkpoint_count": int(book.invalid_checkpoint_count),
        "audit_only_checkpoint_count": int(book.audit_only_checkpoint_count),
        "complete_checkpoint_match_count": int(book.complete_checkpoint_match_count),
        "complete_checkpoint_mismatch_count": int(book.complete_checkpoint_mismatch_count),
        "bootstrap_candidate_count": int(book.bootstrap_candidate_count),
        "bootstrap_count": int(book.bootstrap_count),
        "recovery_count": int(book.recovery_count),
        "bootstrap_bridge_failure_count": int(book.bootstrap_bridge_failure_count),
        "bootstrap_source": book.bootstrap_source,
        "first_bootstrap_ts_recv_ns": book.first_bootstrap_ts_ns,
        "last_bootstrap_ts_recv_ns": book.last_bootstrap_ts_ns,
        "depth_update_count": int(book.depth_count),
        "stale_depth_count": int(book.stale_depth_count),
        "ignored_invalid_depth_count": int(book.ignored_invalid_depth_count),
        "checkpoint_continuation_count": int(book.bootstrap_count),
        "rest_style_bridge_count": 0,
        "median_recovery_ms": float(np.median(book.recovery_latencies_ms)) if book.recovery_latencies_ms else math.nan,
        "p95_recovery_ms": float(np.quantile(book.recovery_latencies_ms, 0.95)) if book.recovery_latencies_ms else math.nan,
        "trade_rows_available": bool(day_files.trades),
        "liquidation_rows": int(len(liq_raw)),
        "price_insert_count": int(book.price_insert_count),
        "price_modify_count": int(book.price_modify_count),
        "price_delete_count": int(book.price_delete_count),
        "price_noop_count": int(book.price_noop_count),
        "bid_level_update_count": int(book.bid_update_count),
        "ask_level_update_count": int(book.ask_update_count),
        "max_bid_levels": int(book.max_bid_levels),
        "max_ask_levels": int(book.max_ask_levels),
        "depth_stream_batch_size": int(DEPTH_STREAM_BATCH_SIZE),
        "snapshot_stream_batch_size": int(SNAPSHOT_STREAM_BATCH_SIZE),
        "runtime_schema_seconds": float(schema_seconds),
        "runtime_trade_aggregation_seconds": float(trade_aggregation_seconds),
        "runtime_liquidation_aggregation_seconds": float(liquidation_aggregation_seconds),
        "runtime_replay_sampling_seconds": float(replay_sampling_seconds),
        "runtime_arrow_depth_decode_seconds": float(fast_stats.arrow_decode_seconds),
        "runtime_compiled_metric_seconds": float(compiled_metric_seconds),
        "runtime_replay_residual_seconds": float(max(0.0, replay_sampling_seconds - fast_stats.arrow_decode_seconds - compiled_metric_seconds)),
        "fast_depth_arrow_batches": int(fast_stats.arrow_batches),
        "fast_depth_rows_decoded": int(fast_stats.rows_decoded),
        "fast_depth_levels_decoded": int(fast_stats.levels_decoded),
        "depth_levels_per_replay_second": float(fast_stats.levels_decoded / replay_sampling_seconds) if replay_sampling_seconds > 0 else math.nan,
        "compiled_replay_engine": "NUMBA_TYPED_DICT_TICK_BOOK",
        "numba_version": str(getattr(_numba, "__version__", "unknown")),
        "price_scale": float(price_scale),
        "runtime_enrichment_seconds": float(enrichment_seconds),
        "runtime_total_build_day_seconds": float(time.perf_counter() - day_started),
        "depth_messages_per_replay_second": float(book.depth_count / replay_sampling_seconds) if replay_sampling_seconds > 0 else math.nan,
    }
    return frame, daily, gap_records, liq_raw, checks


def stress_rows(frame, day_text: str) -> list[dict[str, Any]]:
    np, pd, _, _ = require_data_stack()
    result: list[dict[str, Any]] = []
    metrics = ["trade_quote_volume", "abs_return_1s_bps"]
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        cuts = {q: float(values.quantile(q)) for q in STRESS_QUANTILES}
        strata = [
            ("ALL", np.ones(len(frame), dtype=bool)),
            (">=P50", pd.to_numeric(frame[metric], errors="coerce") >= cuts[0.50]),
            (">=P90", pd.to_numeric(frame[metric], errors="coerce") >= cuts[0.90]),
            (">=P99", pd.to_numeric(frame[metric], errors="coerce") >= cuts[0.99]),
        ]
        for label, mask in strata:
            sub = frame.loc[mask]
            if sub.empty:
                continue
            result.append({
                "receive_date": day_text,
                "stress_metric": metric,
                "stratum": label,
                "threshold": cuts.get({">=P50":0.50,">=P90":0.90,">=P99":0.99}.get(label, -1), math.nan),
                "seconds": len(sub),
                "book_valid_rate": float(sub["book_valid"].mean()),
                "median_state_age_ms": float(pd.to_numeric(sub["state_age_ms"], errors="coerce").median()),
                "force_order_rate": float(sub["force_order_present"].mean()),
            })
    for label, mask in (("FORCE_ORDER", frame["force_order_present"]), ("NO_FORCE_ORDER", ~frame["force_order_present"])):
        sub = frame.loc[mask]
        if not sub.empty:
            result.append({
                "receive_date": day_text,
                "stress_metric": "force_order_present",
                "stratum": label,
                "threshold": math.nan,
                "seconds": len(sub),
                "book_valid_rate": float(sub["book_valid"].mean()),
                "median_state_age_ms": float(pd.to_numeric(sub["state_age_ms"], errors="coerce").median()),
                "force_order_rate": float(sub["force_order_present"].mean()),
            })
    return result


def execution_cost_rows(frame, day_text: str, notionals: Sequence[float]) -> list[dict[str, Any]]:
    np, pd, _, _ = require_data_stack()
    result = []
    valid = frame.loc[frame["book_valid"]].copy()
    for n in notionals:
        tag = str(int(round(n)))
        for side in ("buy", "sell"):
            complete = valid[f"{side}_complete_{tag}"].astype(bool)
            costs = pd.to_numeric(valid.loc[complete, f"{side}_crossing_cost_bps_{tag}"], errors="coerce").dropna()
            result.append({
                "receive_date": day_text,
                "side": side.upper(),
                "quote_notional": n,
                "valid_seconds": len(valid),
                "complete_seconds": int(complete.sum()),
                "complete_rate": float(complete.mean()) if len(valid) else math.nan,
                "median_crossing_cost_bps": float(costs.median()) if len(costs) else math.nan,
                "p90_crossing_cost_bps": float(costs.quantile(0.90)) if len(costs) else math.nan,
                "p95_crossing_cost_bps": float(costs.quantile(0.95)) if len(costs) else math.nan,
                "p99_crossing_cost_bps": float(costs.quantile(0.99)) if len(costs) else math.nan,
            })
    return result


def liquidation_profile_rows(liq, frame, day_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    np, pd, _, _ = require_data_stack()
    if liq.empty:
        return ([{"receive_date": day_text, "liquidation_rows": 0}], [])
    sec_counts = liq.groupby("exchange_second", dropna=True).size() if "exchange_second" in liq else pd.Series(dtype=int)
    obs = pd.to_numeric(liq["observed_selected_notional"], errors="coerce").dropna()
    timing_rows = []
    for source_col in ("exchange_event_position_ms", "trade_time_position_ms"):
        vals = pd.to_numeric(liq[source_col], errors="coerce").dropna()
        if len(vals):
            timing_rows.append({
                "receive_date": day_text,
                "clock": source_col,
                "rows": len(vals),
                "mean_ms": float(vals.mean()),
                "median_ms": float(vals.median()),
                "p10_ms": float(vals.quantile(0.10)),
                "p90_ms": float(vals.quantile(0.90)),
            })
    prof = {
        "receive_date": day_text,
        "liquidation_rows": len(liq),
        "sell_rows": int((liq.get("side", "").astype(str).str.upper() == "SELL").sum()),
        "buy_rows": int((liq.get("side", "").astype(str).str.upper() == "BUY").sum()),
        "exchange_seconds_with_liquidation": int(len(sec_counts)),
        "max_rows_same_exchange_second": int(sec_counts.max()) if len(sec_counts) else 0,
        "exchange_seconds_gt1_row": int((sec_counts > 1).sum()) if len(sec_counts) else 0,
        "median_observed_selected_notional": float(obs.median()) if len(obs) else math.nan,
        "p95_observed_selected_notional": float(obs.quantile(0.95)) if len(obs) else math.nan,
        "selection_semantics": "SELECTED_CENSORED_SNAPSHOT; NOT_TOTAL_LIQUIDATION_VOLUME",
    }
    # Descriptive association only, not a proof of largest-vs-latest selection.
    if len(liq):
        x = liq[["receive_second", "observed_selected_notional"]].copy()
        per_sec = x.groupby("receive_second")["observed_selected_notional"].max().reset_index()
        joined = per_sec.merge(frame[["receive_second", "trade_quote_volume", "abs_return_1s_bps"]], on="receive_second", how="left")
        for metric in ("trade_quote_volume", "abs_return_1s_bps"):
            good = joined[["observed_selected_notional", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            prof[f"spearman_selected_notional_vs_{metric}"] = float(good.corr(method="spearman").iloc[0,1]) if len(good) >= 3 else math.nan
    return [prof], timing_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        atomic_write_text(path, "")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    tmp = short_tmp(path, "csv")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_findings(summary: Mapping[str, Any], daily_rows: Sequence[Mapping[str, Any]], liq_rows: Sequence[Mapping[str, Any]]) -> str:
    valid_rates = [float(r["book_valid_rate"]) for r in daily_rows if r.get("book_valid_rate") is not None]
    gaps = sum(int(r.get("sequence_gap_count", 0)) for r in daily_rows)
    liq_total = sum(int(r.get("liquidation_rows", 0)) for r in liq_rows)
    lines = [
        "# Phase 6A — Execution foundation findings",
        "",
        f"- Builder version: `{BUILDER_VERSION}`",
        f"- Status: **{summary['status']}**",
        f"- Symbol: `{summary['symbol']}`",
        f"- Dates completed: **{summary['completed_dates']}**",
        f"- Output seconds: **{summary['output_rows']:,}**",
        f"- Genuine pu-sequence gaps: **{gaps:,}**",
        f"- forceOrder rows profiled: **{liq_total:,}**",
        "",
        "## Interpretation contract",
        "",
        "This phase does not claim any trading edge. Historical reconstruction bootstraps from a complete stored LocalBook checkpoint, requires exact `pu == snapshot.last_u`, then treats the continuous diff-depth stream as authoritative. Later periodic LocalBook snapshots are audit-only.",
        "",
        "forceOrder observed size is treated as a selected/censored snapshot field. It is **not** summed or calibrated as true liquidation volume.",
        "",
    ]
    if valid_rates:
        lines += [
            "## Book quality",
            "",
            f"Median daily valid-book rate: **{100.0 * sorted(valid_rates)[len(valid_rates)//2]:.3f}%**.",
            "",
            "Use the stress-quality report before any event research: severe periods can have different reconstruction quality than ordinary periods.",
            "",
        ]
    lines += [
        "## Next use",
        "",
        "Phase 6B should consume only rows with `book_valid = true`, carry the quality flags into every event, and calculate gross versus executable/net markouts rather than raw future returns alone.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    numba_version = require_acceleration_stack()
    # Regression for v1.0.2 historical replay contract under compiled state.
    gaps: list[GapRecord] = []
    b = LocalBook()
    complete = {
        "ts_recv_ns": 1_000_000_000,
        "last_update_id": 100,
        "bids": [{"price":"99.0","quantity":"10"},{"price":"98.0","quantity":"10"}],
        "asks": [{"price":"101.0","quantity":"10"},{"price":"102.0","quantity":"10"}],
    }
    # Empty/partial periodic snapshots cannot bootstrap.
    b.apply_snapshot({"ts_recv_ns":900_000_000,"last_update_id":90,
                      "bids":[{"price":"99","quantity":"0"}],
                      "asks":[{"price":"101","quantity":"0"}]})
    assert not b.valid and not b.bridge_pending and b.empty_checkpoint_count == 1

    # Complete checkpoint becomes a candidate, but is not execution-valid until
    # exact local continuation is proved by the first following diff.
    b.apply_snapshot(complete)
    assert not b.valid and b.bridge_pending and b.bootstrap_candidate_count == 1
    pre = b.metrics(1, (100.0,), 0.0, 2000, "BTCUSDT")
    assert not pre["book_valid"] and not pre["sequence_valid"]
    b.apply_depth({
        "ts_recv_ns":1_050_000_000,"first_update_id":101,"final_update_id":101,"previous_final_update_id":100,
        "bid_updates":[{"price":"99.2","quantity":"4"}],"ask_updates":[]
    }, gaps, "1970-01-01")
    assert b.valid and not b.bridge_pending and b.previous_u == 101
    assert b.bootstrap_count == 1 and b.last_bridge_mode == "LOCAL_CHECKPOINT_CONTINUATION"

    # A later malformed snapshot is audit-only and must not destroy state.
    prev_u = b.previous_u
    before_bid = b.bids.best()[0]
    b.apply_snapshot({"ts_recv_ns":1_060_000_000,"last_update_id":101,
                      "bids":[{"price":"999","quantity":"0"}],
                      "asks":[{"price":"1","quantity":"0"}]})
    assert b.valid and b.previous_u == prev_u and b.bids.best()[0] == before_bid
    assert b.audit_only_checkpoint_count == 1

    # Absolute update and deletion on the authoritative diff chain.
    b.apply_depth({
        "ts_recv_ns":1_100_000_000,"first_update_id":102,"final_update_id":103,"previous_final_update_id":101,
        "bid_updates":[],"ask_updates":[{"price":"101.0","quantity":"0"},{"price":"101.5","quantity":"3"}]
    }, gaps, "1970-01-01")
    assert b.valid and b.previous_u == 103 and 1010 not in b.asks.qty

    # A complete later checkpoint is audit-only too; it cannot re-anchor state.
    audit = dict(complete)
    audit["ts_recv_ns"] = 1_150_000_000
    audit["last_update_id"] = 103
    b.apply_snapshot(audit)
    assert b.valid and b.previous_u == 103 and b.audit_only_checkpoint_count == 2

    # Genuine pu gap invalidates; subsequent diffs are ignored until another
    # complete checkpoint plus exact local handshake recovers the replay.
    b.apply_depth({
        "ts_recv_ns":1_200_000_000,"first_update_id":104,"final_update_id":105,"previous_final_update_id":999,
        "bid_updates":[],"ask_updates":[]
    }, gaps, "1970-01-01")
    assert not b.valid and b.actual_sequence_gap_count == 1 and gaps[-1].reason == "PU_SEQUENCE_GAP"
    ignored_before = b.ignored_invalid_depth_count
    b.apply_depth({
        "ts_recv_ns":1_210_000_000,"first_update_id":106,"final_update_id":106,"previous_final_update_id":105,
        "bid_updates":[],"ask_updates":[]
    }, gaps, "1970-01-01")
    assert b.ignored_invalid_depth_count == ignored_before + 1

    recover = dict(complete)
    recover["ts_recv_ns"] = 1_300_000_000
    recover["last_update_id"] = 200
    b.apply_snapshot(recover)
    assert b.bridge_pending and not b.valid
    b.apply_depth({
        "ts_recv_ns":1_310_000_000,"first_update_id":201,"final_update_id":201,"previous_final_update_id":200,
        "bid_updates":[],"ask_updates":[]
    }, gaps, "1970-01-01")
    assert b.valid and b.recovery_count == 1 and b.bootstrap_count == 2

    # A wrong bridge rejects the candidate rather than accepting REST-style overlap.
    c = LocalBook()
    c.apply_snapshot(complete)
    c.apply_depth({
        "ts_recv_ns":1_050_000_000,"first_update_id":99,"final_update_id":101,"previous_final_update_id":98,
        "bid_updates":[],"ask_updates":[]
    }, [], "1970-01-01")
    assert not c.valid and not c.bridge_pending and c.bootstrap_bridge_failure_count == 1

    # Compiled side semantics: quantities are absolute; zero removes the level.
    side = CompiledBookSide(True)
    side.reset([{"price":"100","quantity":"2"}])
    side.update(100.0, 3.0)
    assert abs(float(side.qty[1000]) - 3.0) < 1e-12
    side.update(100.0, 0.0)
    assert 1000 not in side.qty

    # Chunked price order and one-pass metric semantics.
    asks = CompiledBookSide(False)
    asks.reset([
        {"price":"101","quantity":"1"},
        {"price":"102","quantity":"2"},
        {"price":"103","quantity":"3"},
    ])
    depths, execs = asks.scan_depth_and_execution(100.0, (100.0, 250.0), (50.0, 250.0))
    assert list(p for p, _ in asks.iter_best()) == [101.0, 102.0, 103.0]
    assert abs(depths[100.0] - 101.0) < 1e-12
    assert abs(depths[250.0] - (101.0 + 204.0)) < 1e-12
    v50, q50, _, c50 = execs[50.0]
    assert c50 and abs(q50 - 50.0) < 1e-12 and abs(v50 - 101.0) < 1e-12
    _, q250, _, c250 = execs[250.0]
    assert c250 and abs(q250 - 250.0) < 1e-12

    # Randomized compiled-vs-v1.0.3 reference parity on aligned BTCUSDT ticks.
    rng = _np.random.default_rng(20260813)
    ref = LegacyBookSide(False)
    fast = CompiledBookSide(False)
    base_ticks = _np.arange(600_000, 600_500, dtype=_np.int64)
    base_levels = [{"price": str(t / 10.0), "quantity": str(1.0 + (t % 7))} for t in base_ticks]
    ref.reset(base_levels)
    fast.reset(base_levels)
    for _ in range(20):
        ticks = rng.integers(599_800, 600_800, size=500, dtype=_np.int64)
        qty = rng.random(500) * 5.0
        qty[rng.random(500) < 0.2] = 0.0
        for t, q in zip(ticks, qty):
            ref.update(float(t) / 10.0, float(q))
        fast.update_arrays(ticks, qty)
    assert ref.best() == fast.best()
    ref_levels = list(ref.iter_best())
    fast_levels = list(fast.iter_best())
    assert len(ref_levels) == len(fast_levels)
    for (rp, rq), (fp, fq) in zip(ref_levels, fast_levels):
        assert abs(rp - fp) < 1e-12 and abs(rq - fq) < 1e-12
    rd, re = ref.scan_depth_and_execution(60_000.0, DEPTH_BANDS_BPS, DEFAULT_NOTIONALS)
    fd, fe = fast.scan_depth_and_execution(60_000.0, DEPTH_BANDS_BPS, DEFAULT_NOTIONALS)
    for k in rd:
        assert abs(rd[k] - fd[k]) < 1e-6
    for k in re:
        for a, bval in zip(re[k][:3], fe[k][:3]):
            if math.isnan(a) and math.isnan(bval):
                continue
            assert abs(a - bval) < 1e-6
        assert re[k][3] == fe[k][3]

    # Exercise block split/merge without losing deterministic order.
    cp = ChunkedSortedPrices(target=32)
    for x in range(200):
        cp.add(float(x))
    for x in range(0, 200, 3):
        cp.discard(float(x))
    expected = [float(x) for x in range(200) if x % 3 != 0]
    assert list(cp.iter_asc()) == expected
    assert list(cp.iter_desc()) == list(reversed(expected))

    print("SELF-TEST PASS")
    print(f"Compiled replay engine: Numba {numba_version}")
    print("Validated v1.0.2 replay semantics plus Numba tick-book mutation, compiled depth/execution scans, zero-quantity deletion, and the accelerated replay dependency.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase-6A perps execution/data-quality foundation.")
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--max-dates", type=int)
    p.add_argument("--notionals", default=",".join(str(int(x)) for x in DEFAULT_NOTIONALS), help="Comma-separated quote notionals.")
    p.add_argument("--taker-fee-bps", type=float, default=0.0, help="Optional user-specific taker-fee assumption. Raw crossing cost remains separate.")
    p.add_argument("--max-book-age-ms", type=int, default=DEFAULT_MAX_BOOK_AGE_MS)
    p.add_argument(
        "--price-scale", type=float, default=DEFAULT_PRICE_SCALE,
        help="Integer price ticks per USD for compiled replay. BTCUSDT default 10 = 0.1 USDT tick; every depth price is alignment-validated.",
    )
    p.add_argument(
        "--progress-every-depth",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_DEPTH,
        help="Print replay progress every N depth messages; use 0 to disable.",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    np, pd, _, pq = require_data_stack()
    numba_version = require_acceleration_stack()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    notionals = tuple(sorted({float(x.strip()) for x in str(args.notionals).split(",") if x.strip()}))
    if not notionals or any((not math.isfinite(x) or x <= 0) for x in notionals):
        raise SystemExit("--notionals must contain positive finite quote notionals")
    if args.max_book_age_ms <= 0:
        raise SystemExit("--max-book-age-ms must be positive")
    if not math.isfinite(args.price_scale) or args.price_scale <= 0:
        raise SystemExit("--price-scale must be positive and finite")
    if args.progress_every_depth < 0:
        raise SystemExit("--progress-every-depth must be >= 0")

    jit_warmup_seconds = warm_compiled_kernels(args.price_scale)

    days = discover_days(args.canonical_root, start, end, args.max_dates)
    if not days:
        raise SystemExit(f"No days with both perps/depth_updates and perps/book_snapshots found under {args.canonical_root}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    metadata_root = args.output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    conn = init_state(metadata_root / "build_state.sqlite")

    config = {
        "builder_version": BUILDER_VERSION,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "resumability": "per-day parquet plus signed quality sidecar plus SQLite state",
        "symbol": args.symbol,
        "canonical_root": str(args.canonical_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "report_root": str(args.report_root.resolve()),
        "start_date": days[0].day.isoformat(),
        "end_date": days[-1].day.isoformat(),
        "notionals_quote": list(notionals),
        "depth_bands_bps": list(DEPTH_BANDS_BPS),
        "taker_fee_bps_assumption": args.taker_fee_bps,
        "max_book_age_ms": args.max_book_age_ms,
        "causal_clock": "ts_recv_ns",
        "force_order_size_semantics": "SELECTED_CENSORED_SNAPSHOT_NOT_TOTAL_VOLUME",
        "sequence_contract": (
            "complete stored LocalBook checkpoint may bootstrap/recover only when first following diff "
            "satisfies pu==snapshot_last_update_id; thereafter diffs are authoritative and require "
            "pu==previous_u; later periodic snapshots are audit-only; invalidate on genuine gap/cross"
        ),
        "historical_bootstrap_source": "COMPLETE_LOCALBOOK_CHECKPOINT_SURROGATE",
        "raw_binance_rest_snapshot_levels_persisted": False,
        "performance_contract": "semantics-preserving v1.0.4: Arrow-native nested numeric decode + Numba typed-dict tick replay + compiled tick scan metrics",
        "compiled_replay_engine": "NUMBA_TYPED_DICT_TICK_BOOK",
        "numba_version": numba_version,
        "numba_jit_warmup_seconds": jit_warmup_seconds,
        "price_scale": args.price_scale,
        "depth_stream_batch_size": DEPTH_STREAM_BATCH_SIZE,
        "snapshot_stream_batch_size": SNAPSHOT_STREAM_BATCH_SIZE,
        "progress_every_depth": args.progress_every_depth,
    }
    atomic_write_json(metadata_root / "run_config.json", config)

    print("Building Phase-6A BTCUSDT execution/data-quality foundation...")
    print(f"Builder version   : {BUILDER_VERSION}")
    print(f"Run ID            : {run_id}")
    print(f"Symbol            : {args.symbol}")
    print(f"Canonical root    : {args.canonical_root}")
    print(f"Output root       : {args.output_root}")
    print(f"Report root       : {args.report_root}")
    print(f"Dates             : {len(days)} ({days[0].day} to {days[-1].day})")
    print(f"Quote notionals   : {', '.join(f'${x:,.0f}' for x in notionals)}")
    print(f"Taker fee assumed : {args.taker_fee_bps:g} bps (raw crossing cost also retained)")
    print("forceOrder size   : selected/censored observation; observed sums are NOT total liquidation volume")
    print("Book replay        : complete local checkpoint bootstrap -> authoritative diffs; later snapshots audit-only")
    print(f"Performance        : Arrow-native decode + Numba compiled tick replay (numba {numba_version}, scale={args.price_scale:g})")
    print(f"JIT warm-up        : {jit_warmup_seconds:.3f}s (excluded from per-day timing)")

    daily_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    stress_all: list[dict[str, Any]] = []
    costs_all: list[dict[str, Any]] = []
    liq_all: list[dict[str, Any]] = []
    liq_timing_all: list[dict[str, Any]] = []
    checks_all: list[Check] = []
    completed = skipped = failed = output_rows = 0

    for i, dfiles in enumerate(days, 1):
        sig_payload = {
            "files": file_signature(dfiles.all_files),
            "symbol": args.symbol,
            "notionals": notionals,
            "fee": args.taker_fee_bps,
            "max_age": args.max_book_age_ms,
            "price_scale": args.price_scale,
            "version": BUILDER_VERSION,
        }
        sig = sha256_text(compact_json(sig_payload))
        out = output_path(args.output_root, args.symbol, dfiles.day)
        sidecar = quality_sidecar_path(args.output_root, args.symbol, dfiles.day)
        if not args.force and can_reuse(conn, dfiles.day, sig, out, sidecar):
            frame = pd.read_parquet(out)
            print(f"[{i:02d}/{len(days):02d}] {dfiles.day}: COMPLETED_REUSED ({len(frame):,} seconds)")
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            daily = dict(saved["daily"])
            daily["reused"] = True
            daily_rows.append(daily)
            gap_rows.extend(saved.get("gaps", []))
            liq_all.extend(saved.get("liquidation_profile", []))
            liq_timing_all.extend(saved.get("liquidation_timing", []))
            checks_all.extend(Check(**x) for x in saved.get("checks", []))
            stress_all.extend(stress_rows(frame, dfiles.day.isoformat()))
            costs_all.extend(execution_cost_rows(frame, dfiles.day.isoformat(), notionals))
            completed += 1
            skipped += 1
            output_rows += len(frame)
            continue
        try:
            frame, daily, gaps, liq_raw, checks = build_day(
                dfiles, args.symbol, notionals, args.taker_fee_bps, args.max_book_age_ms, args.progress_every_depth, args.price_scale
            )
            metadata = {
                "execution_foundation_builder_version": BUILDER_VERSION,
                "execution_foundation_schema_version": OUTPUT_SCHEMA_VERSION,
                "symbol": args.symbol,
                "receive_date": dfiles.day.isoformat(),
                "input_signature": sig,
                "causal_clock": "ts_recv_ns",
                "force_order_size_semantics": "SELECTED_CENSORED_SNAPSHOT_NOT_TOTAL_VOLUME",
                "snapshot_handshake_contract": "COMPLETE_LOCAL_CHECKPOINT_THEN_EXACT_PU_CONTINUATION",
                "periodic_snapshot_state_role": "AUDIT_ONLY_AFTER_BOOTSTRAP",
                "compiled_replay_engine": "NUMBA_TYPED_DICT_TICK_BOOK",
                "price_scale": args.price_scale,
            }
            atomic_write_parquet(out, frame, metadata)
            lp, lt = liquidation_profile_rows(liq_raw, frame, dfiles.day.isoformat())
            sidecar_payload = {
                "builder_version": BUILDER_VERSION,
                "input_signature": sig,
                "receive_date": dfiles.day.isoformat(),
                "daily": daily,
                "gaps": [asdict(x) for x in gaps],
                "liquidation_profile": lp,
                "liquidation_timing": lt,
                "checks": [asdict(x) for x in checks],
            }
            atomic_write_json(sidecar, sidecar_payload)
            mark_state(conn, dfiles.day, sig, "COMPLETED", len(frame), out, run_id)
            daily_rows.append(daily)
            gap_rows.extend(asdict(x) for x in gaps)
            stress_all.extend(stress_rows(frame, dfiles.day.isoformat()))
            costs_all.extend(execution_cost_rows(frame, dfiles.day.isoformat(), notionals))
            liq_all.extend(lp)
            liq_timing_all.extend(lt)
            checks_all.extend(checks)
            completed += 1
            output_rows += len(frame)
            print(
                f"[{i:02d}/{len(days):02d}] {dfiles.day}: COMPLETED "
                f"({len(frame):,} seconds, gaps={len(gaps):,}, "
                f"valid={100*daily['book_valid_rate']:.2f}%, "
                f"build={daily['runtime_total_build_day_seconds']:.1f}s)"
            )
        except Exception as exc:
            failed += 1
            mark_state(conn, dfiles.day, sig, "FAILED", 0, out, run_id, f"{type(exc).__name__}: {exc}")
            checks_all.append(Check("day", dfiles.day.isoformat(), "FAIL", f"{type(exc).__name__}: {exc}", "successful build"))
            print(f"[{i:02d}/{len(days):02d}] {dfiles.day}: FAIL - {type(exc).__name__}: {exc}")

    # Global invariants on freshly/fully available outputs.
    checks_all.append(Check("run", "failed_dates", "PASS" if failed == 0 else "FAIL", failed, 0))
    checks_all.append(Check("run", "output_rows", "PASS" if output_rows > 0 else "FAIL", output_rows, ">0"))
    for drow in daily_rows:
        if int(drow.get("rows", 0)) != SECONDS_PER_DAY:
            checks_all.append(Check("output", f"{drow['receive_date']}/seconds_per_day", "FAIL", drow.get("rows"), SECONDS_PER_DAY))
    # forceOrder throttle observation is reported, not enforced: historical semantics may vary.
    fail_checks = [c for c in checks_all if c.status == "FAIL"]
    status = "PASS" if not fail_checks and failed == 0 else "FAIL"

    write_csv(args.report_root / "execution_foundation_daily_quality.csv", daily_rows)
    write_csv(args.report_root / "execution_foundation_sequence_gaps.csv", gap_rows)
    write_csv(args.report_root / "execution_foundation_stress_quality.csv", stress_all)
    write_csv(args.report_root / "execution_foundation_execution_costs.csv", costs_all)
    write_csv(args.report_root / "execution_foundation_liquidation_profile.csv", liq_all)
    write_csv(args.report_root / "execution_foundation_liquidation_timing.csv", liq_timing_all)
    performance_keys = (
        "receive_date", "runtime_total_build_day_seconds", "runtime_schema_seconds",
        "runtime_trade_aggregation_seconds", "runtime_liquidation_aggregation_seconds",
        "runtime_replay_sampling_seconds", "runtime_arrow_depth_decode_seconds",
        "runtime_compiled_metric_seconds", "runtime_replay_residual_seconds", "runtime_enrichment_seconds",
        "depth_messages_per_replay_second", "depth_levels_per_replay_second", "depth_update_count",
        "fast_depth_arrow_batches", "fast_depth_rows_decoded", "fast_depth_levels_decoded",
        "bid_level_update_count", "ask_level_update_count",
        "price_insert_count", "price_modify_count", "price_delete_count", "price_noop_count",
        "max_bid_levels", "max_ask_levels", "compiled_replay_engine", "numba_version", "price_scale",
        "depth_stream_batch_size", "snapshot_stream_batch_size",
    )
    performance_rows = [{k: row.get(k) for k in performance_keys} for row in daily_rows]
    write_csv(args.report_root / "execution_foundation_performance.csv", performance_rows)
    write_csv(args.report_root / "execution_foundation_invariants.csv", [asdict(c) for c in checks_all])

    summary = {
        "builder_version": BUILDER_VERSION,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "resumability": "per-day parquet plus signed quality sidecar plus SQLite state",
        "status": status,
        "symbol": args.symbol,
        "selected_dates": len(days),
        "completed_dates": completed,
        "reused_dates": skipped,
        "failed_dates": failed,
        "output_rows": output_rows,
        "failed_invariants": len(fail_checks),
        "first_date": days[0].day.isoformat(),
        "last_date": days[-1].day.isoformat(),
        "notionals_quote": list(notionals),
        "taker_fee_bps_assumption": args.taker_fee_bps,
        "force_order_size_semantics": "SELECTED_CENSORED_SNAPSHOT_NOT_TOTAL_VOLUME",
        "completed_at": utc_now_iso(),
    }
    atomic_write_json(metadata_root / "latest_summary.json", summary)
    atomic_write_text(args.report_root / "execution_foundation_findings.md", build_findings(summary, daily_rows, liq_all))

    print("\nPhase-6A execution foundation summary")
    print("------------------------------------")
    print(f"Status                 : {status}")
    print(f"Dates completed        : {completed}/{len(days)}")
    print(f"Output seconds         : {output_rows:,}")
    print(f"Failed dates           : {failed}")
    print(f"Failed invariants      : {len(fail_checks)}")
    if daily_rows:
        rates = [float(x["book_valid_rate"]) for x in daily_rows if x.get("book_valid_rate") is not None]
        if rates:
            print(f"Median daily book valid: {100*np.median(rates):.3f}%")
    print(f"Summary                : {metadata_root / 'latest_summary.json'}")
    print(f"Findings               : {args.report_root / 'execution_foundation_findings.md'}")
    print("No alpha or strategy claim is made by Phase 6A.")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
