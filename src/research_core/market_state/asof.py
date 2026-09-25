"""Integer-nanosecond as-of selection primitives."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

from research_core.clocks.contract import require_int64_ns


@dataclass(frozen=True)
class AsofSelection:
    index: int
    ts_ns: int
    age_ns: int
    tie_ambiguous: bool


def latest_prior_index(
    ts_ns: Sequence[int],
    cutoff_ts_ns: object,
    *,
    deterministic_tie_keys: Sequence[object] | None = None,
) -> AsofSelection | None:
    """Select the latest row with receive timestamp <= cutoff.

    ``ts_ns`` must already be ordered by receive timestamp and, when equal
    timestamps exist, by canonical tie keys.  If no tie keys are supplied and
    the latest timestamp has multiple rows, the selection is still the latest
    timestamp but is explicitly marked ambiguous so callers may suppress any
    order-sensitive value such as a single "latest price".
    """

    cutoff = require_int64_ns(cutoff_ts_ns, "cutoff_ts_ns")
    if deterministic_tie_keys is not None and len(deterministic_tie_keys) != len(ts_ns):
        raise ValueError("deterministic_tie_keys length must match ts_ns")
    if not ts_ns:
        return None
    for value in ts_ns:
        require_int64_ns(value, "ts_ns")
    i = bisect_right(ts_ns, cutoff) - 1
    if i < 0:
        return None
    ts = int(ts_ns[i])
    lo = i
    while lo > 0 and int(ts_ns[lo - 1]) == ts:
        lo -= 1
    ambiguous = bool(i > lo and deterministic_tie_keys is None)
    return AsofSelection(index=i, ts_ns=ts, age_ns=cutoff - ts, tie_ambiguous=ambiguous)


def assert_available_by_cutoff(source_ts_ns: object, cutoff_ts_ns: object) -> tuple[int, int]:
    source = require_int64_ns(source_ts_ns, "source_ts_ns")
    cutoff = require_int64_ns(cutoff_ts_ns, "cutoff_ts_ns")
    if source > cutoff:
        raise ValueError("source row occurs after causal cutoff")
    return source, cutoff
