"""Explicit execution-latency assumptions for R0.6.

R0.6 does not pretend to know live production latency.  Every component is an
integer number of nanoseconds and must be supplied explicitly (or deliberately
set to zero for the infrastructure-only idealized audit profile).
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from research_core.clocks.contract import require_int64_ns


def _nonnegative_ns(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer number of nanoseconds")
    iv = int(value)
    if iv < 0:
        raise ValueError(f"{field} must be >= 0")
    return iv


@dataclass(frozen=True)
class LatencyAssumptions:
    computation_ns: int
    decision_ns: int
    order_submission_ns: int
    network_ns: int
    exchange_processing_ns: int
    fill_completion_ns: int
    profile_id: str = "EXPLICIT"

    def __post_init__(self) -> None:
        for name in (
            "computation_ns", "decision_ns", "order_submission_ns",
            "network_ns", "exchange_processing_ns", "fill_completion_ns",
        ):
            object.__setattr__(self, name, _nonnegative_ns(getattr(self, name), name))
        if not str(self.profile_id).strip():
            raise ValueError("profile_id must be non-empty")


ZERO_LATENCY_AUDIT = LatencyAssumptions(0, 0, 0, 0, 0, 0, "ZERO_LATENCY_AUDIT_ONLY")


def build_execution_timeline(information_available_ts_ns: object, assumptions: LatencyAssumptions) -> dict[str, int | str]:
    info = require_int64_ns(information_available_ts_ns, "information_available_ts_ns")
    computation_complete = info + assumptions.computation_ns
    decision = computation_complete + assumptions.decision_ns
    order_send = decision + assumptions.order_submission_ns
    arrival = order_send + assumptions.network_ns
    executable = arrival + assumptions.exchange_processing_ns
    fill_complete = executable + assumptions.fill_completion_ns
    return {
        "latency_profile_id": assumptions.profile_id,
        "information_available_ts_ns": info,
        "computation_complete_ts_ns": computation_complete,
        "decision_ts_ns": decision,
        "order_send_ts_ns": order_send,
        "estimated_exchange_arrival_ts_ns": arrival,
        "executable_entry_ts_ns": executable,
        "fill_complete_ts_ns": fill_complete,
    }
