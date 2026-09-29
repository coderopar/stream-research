"""Explicit trading-friction accounting for R0.6."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostAssumptions:
    """Cost inputs are assumptions, never inferred from future performance.

    ``funding_bps`` and ``other_bps`` are signed: a negative value represents a
    credit.  Fees must be non-negative.
    """

    taker_fee_bps: float
    funding_bps: float
    other_bps: float
    assumption_id: str = "EXPLICIT"

    def __post_init__(self) -> None:
        vals = (self.taker_fee_bps, self.funding_bps, self.other_bps)
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("all cost assumptions must be finite")
        if float(self.taker_fee_bps) < 0:
            raise ValueError("taker_fee_bps must be >= 0")
        if not str(self.assumption_id).strip():
            raise ValueError("assumption_id must be non-empty")


def total_cost_bps(raw_crossing_cost_bps: float | None, assumptions: CostAssumptions) -> float | None:
    if raw_crossing_cost_bps is None:
        return None
    x = float(raw_crossing_cost_bps)
    if not math.isfinite(x):
        return None
    return x + float(assumptions.taker_fee_bps) + float(assumptions.funding_bps) + float(assumptions.other_bps)
