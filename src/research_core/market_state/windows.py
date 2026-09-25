"""Half-open receive-time trade bucket primitives for R0.5."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Sequence

from research_core.clocks.contract import NS_PER_SECOND, require_int64_ns


@dataclass(frozen=True)
class TradeBucket:
    start_ts_ns: int
    end_ts_ns: int
    trade_count: int
    quantity_sum: float
    notional_sum: float
    buyer_maker_true_quantity: float
    buyer_maker_false_quantity: float
    buyer_maker_unknown_quantity: float
    min_trade_price: float | None
    max_trade_price: float | None
    first_trade_ts_ns: int | None
    last_trade_ts_ns: int | None


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    # pandas/numpy booleans stringify safely here; numeric 0/1 are allowed.
    if value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1"}:
        return True
    if text in {"false", "f", "0"}:
        return False
    return None


def summarize_trade_bucket(
    *,
    trade_ts_ns: Sequence[int],
    trade_prices: Sequence[float],
    trade_quantities: Sequence[float],
    buyer_is_maker: Sequence[object],
    start_ts_ns: object,
    end_ts_ns: object,
) -> TradeBucket:
    """Aggregate observed trades in the exact half-open interval [start, end)."""

    n = len(trade_ts_ns)
    if not (len(trade_prices) == len(trade_quantities) == len(buyer_is_maker) == n):
        raise ValueError("trade arrays must have equal lengths")
    start = require_int64_ns(start_ts_ns, "start_ts_ns")
    end = require_int64_ns(end_ts_ns, "end_ts_ns")
    if end <= start:
        raise ValueError("trade bucket end must be > start")
    lo = bisect_left(trade_ts_ns, start)
    hi = bisect_left(trade_ts_ns, end)
    count = 0
    qty_sum = 0.0
    notional_sum = 0.0
    bm_true = 0.0
    bm_false = 0.0
    bm_unknown = 0.0
    min_px: float | None = None
    max_px: float | None = None
    first_ts: int | None = None
    last_ts: int | None = None
    for ts, px, qty, maker in zip(
        trade_ts_ns[lo:hi],
        trade_prices[lo:hi],
        trade_quantities[lo:hi],
        buyer_is_maker[lo:hi],
    ):
        ts_i = require_int64_ns(ts, "trade_ts_ns")
        px_f = float(px)
        qty_f = float(qty)
        if not (isfinite(px_f) and px_f > 0 and isfinite(qty_f) and qty_f > 0):
            continue
        count += 1
        qty_sum += qty_f
        notional_sum += px_f * qty_f
        min_px = px_f if min_px is None else min(min_px, px_f)
        max_px = px_f if max_px is None else max(max_px, px_f)
        first_ts = ts_i if first_ts is None else min(first_ts, ts_i)
        last_ts = ts_i if last_ts is None else max(last_ts, ts_i)
        maker_b = _bool_or_none(maker)
        if maker_b is True:
            bm_true += qty_f
        elif maker_b is False:
            bm_false += qty_f
        else:
            bm_unknown += qty_f
    return TradeBucket(
        start_ts_ns=start,
        end_ts_ns=end,
        trade_count=count,
        quantity_sum=float(qty_sum),
        notional_sum=float(notional_sum),
        buyer_maker_true_quantity=float(bm_true),
        buyer_maker_false_quantity=float(bm_false),
        buyer_maker_unknown_quantity=float(bm_unknown),
        min_trade_price=min_px,
        max_trade_price=max_px,
        first_trade_ts_ns=first_ts,
        last_trade_ts_ns=last_ts,
    )


def relative_bucket_bounds(
    cutoff_ts_ns: object,
    *,
    lookback_sec: int,
    bucket_sec: int,
) -> list[tuple[int, int, int]]:
    """Return candidate-relative half-open buckets ending exactly at cutoff.

    The returned offset is a negative integer number of seconds from cutoff to
    the bucket start.  Example with lookback=3,bucket=1 -> -3,-2,-1.
    """

    cutoff = require_int64_ns(cutoff_ts_ns, "cutoff_ts_ns")
    if isinstance(lookback_sec, bool) or not isinstance(lookback_sec, Integral) or lookback_sec <= 0:
        raise ValueError("lookback_sec must be a positive integer")
    if isinstance(bucket_sec, bool) or not isinstance(bucket_sec, Integral) or bucket_sec <= 0:
        raise ValueError("bucket_sec must be a positive integer")
    if int(lookback_sec) % int(bucket_sec) != 0:
        raise ValueError("lookback_sec must be divisible by bucket_sec")
    out: list[tuple[int, int, int]] = []
    for offset in range(-int(lookback_sec), 0, int(bucket_sec)):
        start = cutoff + offset * NS_PER_SECOND
        end = start + int(bucket_sec) * NS_PER_SECOND
        out.append((offset, start, end))
    return out
