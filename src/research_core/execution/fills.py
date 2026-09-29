"""Marketable-order fill mechanics for a reconstructed visible order book."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class FillResult:
    side: Side
    requested_quote: float
    filled_quote: float
    filled_base: float
    vwap: float | None
    complete: bool
    raw_crossing_cost_bps: float | None


def simulate_marketable_fill(
    *,
    side: Side,
    quote_notional: float,
    levels: Iterable[tuple[float, float]],
    arrival_mid: float,
) -> FillResult:
    side = str(side).upper()  # type: ignore[assignment]
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    target = float(quote_notional)
    mid = float(arrival_mid)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("quote_notional must be positive and finite")
    if not math.isfinite(mid) or mid <= 0:
        raise ValueError("arrival_mid must be positive and finite")

    qdone = 0.0
    bdone = 0.0
    tol = max(1e-9, target * 1e-12)
    for price, quantity in levels:
        p = float(price)
        q = float(quantity)
        if not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q <= 0:
            continue
        available_quote = p * q
        take_quote = min(max(0.0, target - qdone), available_quote)
        if take_quote > 0:
            qdone += take_quote
            bdone += take_quote / p
        if target - qdone <= tol:
            break

    complete = (target - qdone) <= tol
    vwap = qdone / bdone if bdone > 0 else None
    crossing: float | None = None
    if complete and vwap is not None:
        if side == "BUY":
            crossing = (vwap / mid - 1.0) * 10_000.0
        else:
            crossing = (1.0 - vwap / mid) * 10_000.0
    return FillResult(side, target, qdone, bdone, vwap, complete, crossing)
