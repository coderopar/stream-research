"""R0.3 corrected-D0 substrate primitives.

These helpers intentionally separate three concepts that were historically
blurred together:

* the legacy integer ``candidate_second`` used to define candidate identity;
* the recovered exact receive-time anchor, stored as int64 nanoseconds; and
* the outcome interval, which begins only after the causal feature window.

R0.3 is development-data infrastructure.  The zero-latency execution fields in
``build_scientific_label_clock`` exist only so the R0.2 clock contract can audit
scientific label timing.  They are not an execution model and must not be used
for PnL.  R0.6 replaces them with a marketable-order execution simulation.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import isfinite
from numbers import Integral
from typing import Iterable, Mapping, Sequence

from research_core.clocks.contract import (
    D0_END_EXCLUSIVE_NS,
    D0_START_NS,
    NS_PER_SECOND,
    assert_clock_contract,
    require_int64_ns,
)

LEGACY_CANDIDATE_DEFINITION_ID = "LEGACY_7M2_EARLY_25BP_RESET5_RELEASE10_REARM60"
PRIMARY_OBSERVATION_HORIZON_SEC = 3
OUTCOME_HORIZONS_SEC: tuple[int, ...] = (300, 600, 900, 1800)
OUTCOME_TARGETS_BPS: tuple[int, ...] = (35, 50, 75, 100)
CANDIDATE_THRESHOLD_BPS = 25.0
MAX_TRADE_STALE_NS = 5 * NS_PER_SECOND

FUTURE_NAME_TOKENS: tuple[str, ...] = (
    "outcome",
    "future",
    "reach_",
    "reaches_",
    "seconds_to_",
    "mfe",
    "mae",
    "first_passage",
    "p35",
    "p50",
    "p75",
    "p100",
)


def validate_exact_anchor(candidate_second: object, exact_anchor_ts_ns: object) -> int:
    """Validate an exact anchor against the legacy candidate-second envelope.

    ``candidate_second`` remains lineage metadata only.  It is never converted
    into a repaired timestamp.  The exact anchor must already exist as an
    integer source timestamp.
    """

    if isinstance(candidate_second, bool) or not isinstance(candidate_second, Integral):
        raise TypeError("candidate_second must be an integer epoch second")
    anchor = require_int64_ns(exact_anchor_ts_ns, "exact_anchor_ts_ns")
    start = int(candidate_second) * NS_PER_SECOND
    end = (int(candidate_second) + 1) * NS_PER_SECOND
    if anchor < start or anchor >= end:
        raise ValueError(
            f"exact anchor {anchor} lies outside the historical candidate-second envelope "
            f"[{start}, {end})"
        )
    if anchor < D0_START_NS or anchor >= D0_END_EXCLUSIVE_NS:
        raise ValueError("exact anchor is outside frozen D0")
    return anchor



def frozen_candidate_availability_ts_ns(candidate_second: object) -> int:
    """Return the conservative availability time of the frozen 1-second policy.

    Phase 7M.2/7K.1 generated candidates from a one-second price series. The
    selected price for ``candidate_second`` is only frozen once that one-second
    observation has completed.  Therefore the causal availability timestamp is
    the interval-close boundary, not an invented intrasecond raw-trade hit.

    This is a policy-clock boundary, not a reconstructed exact market-event
    timestamp and never substitutes for ``anchor_ts_ns``.
    """

    if isinstance(candidate_second, bool) or not isinstance(candidate_second, Integral):
        raise TypeError("candidate_second must be an integer epoch second")
    return (int(candidate_second) + 1) * NS_PER_SECOND


def legacy_timing_order_valid(
    origin_second: object,
    release_second: object,
    candidate_second: object,
) -> bool:
    """Validate the frozen one-second state-machine ordering.

    Same-second release/candidate transitions are explicitly valid in the
    historical 7K.1 state machine for large one-second jumps, so R0.3 must not
    strengthen ``<=`` into ``<``.
    """

    values = (origin_second, release_second, candidate_second)
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        raise TypeError("origin_second, release_second, and candidate_second must be integers")
    origin = int(origin_second)
    release = int(release_second)
    candidate = int(candidate_second)
    return origin <= release <= candidate


def build_scientific_label_clock(
    exact_anchor_ts_ns: object,
    *,
    candidate_detection_ts_ns: object | None = None,
    observation_horizon_seconds: int = PRIMARY_OBSERVATION_HORIZON_SEC,
) -> dict[str, int]:
    """Build the governed D0 scientific-label clock from an exact anchor.

    Candidate availability and the exact depth anchor are separate clocks.
    For the frozen one-second policy, candidate availability is the deterministic
    interval-close boundary produced by :func:`frozen_candidate_availability_ts_ns`.
    The exact book anchor may occur before or after the physical threshold-crossing
    trade inside that second.  The feature window ends at
    ``anchor + observation_horizon`` and the future label interval begins at
    exactly the same timestamp.

    The execution fields are equal to the decision timestamp only for R0.2
    contract completeness.  They represent *no execution model*.
    """

    if observation_horizon_seconds != PRIMARY_OBSERVATION_HORIZON_SEC:
        raise ValueError(
            f"R0.3 freezes the scientific observation horizon at "
            f"{PRIMARY_OBSERVATION_HORIZON_SEC}s; got {observation_horizon_seconds}"
        )
    anchor = require_int64_ns(exact_anchor_ts_ns, "exact_anchor_ts_ns")
    if candidate_detection_ts_ns is None:
        raise ValueError("candidate_detection_ts_ns is required for governed R0.3 clocks")
    detection = require_int64_ns(candidate_detection_ts_ns, "candidate_detection_ts_ns")
    decision = anchor + observation_horizon_seconds * NS_PER_SECOND
    if detection > decision:
        raise ValueError("candidate is not causally available by the feature-window decision time")
    record = {
        "candidate_detection_ts_ns": detection,
        "anchor_ts_ns": anchor,
        "feature_window_start_ts_ns": anchor,
        "feature_window_end_ts_ns": decision,
        "information_available_ts_ns": decision,
        "decision_ts_ns": decision,
        "order_send_ts_ns": decision,
        "estimated_exchange_arrival_ts_ns": decision,
        "executable_entry_ts_ns": decision,
        "outcome_start_ts_ns": decision,
    }
    assert_clock_contract(record)
    return record


def future_like_columns(columns: Iterable[str]) -> list[str]:
    bad: list[str] = []
    allowed_clock_fields = {"outcome_start_ts_ns"}
    for col in columns:
        name = str(col)
        if name in allowed_clock_fields:
            continue
        lc = name.lower()
        if any(token in lc for token in FUTURE_NAME_TOKENS):
            bad.append(name)
    return sorted(set(bad))


def assert_causal_columns_future_clean(columns: Iterable[str]) -> None:
    bad = future_like_columns(columns)
    if bad:
        raise ValueError(f"future-like columns are forbidden in the R0.3 causal substrate: {bad}")


def directional_bps(price: float, reference: float, direction: int) -> float:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    p = float(price)
    r = float(reference)
    if not (isfinite(p) and isfinite(r) and r > 0 and p > 0):
        return float("nan")
    return float(direction) * (p / r - 1.0) * 10_000.0



def earliest_threshold_crossing_in_second(
    *,
    trade_ts_ns: Sequence[int],
    trade_prices: Sequence[float],
    candidate_second: int,
    origin_price: float,
    direction: int,
    threshold_bps: float = CANDIDATE_THRESHOLD_BPS,
) -> int | None:
    """Return the first exact raw-trade receive timestamp crossing a threshold.

    Search is restricted to the half-open frozen candidate second [T, T+1s).
    Equal receive timestamps are preserved; the earliest timestamp is therefore
    well-defined even when multiple trades share that timestamp.
    """

    if len(trade_ts_ns) != len(trade_prices):
        raise ValueError("trade timestamp and price arrays must have equal length")
    if isinstance(candidate_second, bool) or not isinstance(candidate_second, Integral):
        raise TypeError("candidate_second must be an integer epoch second")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if not isfinite(float(threshold_bps)) or float(threshold_bps) <= 0:
        raise ValueError("threshold_bps must be finite and > 0")
    start = int(candidate_second) * NS_PER_SECOND
    end = (int(candidate_second) + 1) * NS_PER_SECOND
    lo = bisect_left(trade_ts_ns, start)
    hi = bisect_left(trade_ts_ns, end)
    for ts, px in zip(trade_ts_ns[lo:hi], trade_prices[lo:hi]):
        ts_i = require_int64_ns(ts, "trade_ts_ns")
        if directional_bps(float(px), float(origin_price), direction) + 1e-12 >= float(threshold_bps):
            return ts_i
    return None

def latest_prior_trade(
    trade_ts_ns: Sequence[int],
    trade_prices: Sequence[float],
    point_ts_ns: int,
    *,
    max_stale_ns: int = MAX_TRADE_STALE_NS,
) -> tuple[int | None, float | None]:
    """Return the latest trade at or before ``point_ts_ns`` if fresh enough."""

    if len(trade_ts_ns) != len(trade_prices):
        raise ValueError("trade timestamp and price arrays must have equal length")
    point = require_int64_ns(point_ts_ns, "point_ts_ns")
    if max_stale_ns < 0:
        raise ValueError("max_stale_ns must be >= 0")
    if not trade_ts_ns:
        return None, None
    i = bisect_right(trade_ts_ns, point) - 1
    if i < 0:
        return None, None
    ts = require_int64_ns(trade_ts_ns[i], "trade_ts_ns")
    px = float(trade_prices[i])
    if not (isfinite(px) and px > 0):
        return None, None
    if point - ts > max_stale_ns:
        return None, None
    return ts, px



def summarize_outcome_horizons(
    *,
    trade_ts_ns: Sequence[int],
    trade_prices: Sequence[float],
    outcome_start_ts_ns: int,
    decision_price: float,
    origin_price: float,
    direction: int,
    horizons_seconds: Sequence[int],
    targets_bps: Sequence[int] = OUTCOME_TARGETS_BPS,
) -> dict[int, dict[str, object]]:
    """Summarize multiple future horizons in one monotone trade scan.

    This is semantically equivalent to repeated ``summarize_outcome_horizon``
    calls with ``fully_observed=True``, but it avoids rescanning the same raw
    BTC trade path once per horizon. Trades stamped exactly at a horizon end
    remain included, matching the single-horizon helper's ``bisect_right``
    semantics.
    """

    if len(trade_ts_ns) != len(trade_prices):
        raise ValueError("trade timestamp and price arrays must have equal length")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    horizons = sorted(set(int(h) for h in horizons_seconds))
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons_seconds must contain positive integers")

    start = require_int64_ns(outcome_start_ts_ns, "outcome_start_ts_ns")
    dp = float(decision_price)
    op = float(origin_price)
    if not (isfinite(dp) and dp > 0 and isfinite(op) and op > 0):
        raise ValueError("decision_price and origin_price must be finite and > 0")

    target_ids = [int(t) for t in targets_bps]
    target_hit: dict[int, int | None] = {}
    start_origin_dr = directional_bps(dp, op, direction)
    for target in target_ids:
        target_hit[target] = start if start_origin_dr + 1e-12 >= target else None

    idx = bisect_right(trade_ts_ns, start)
    future_count = 0
    max_decision_dr = 0.0
    min_decision_dr = 0.0
    last_ts = start
    last_px = dp
    out: dict[int, dict[str, object]] = {}

    for horizon in horizons:
        end = start + horizon * NS_PER_SECOND
        while idx < len(trade_ts_ns):
            ts = require_int64_ns(trade_ts_ns[idx], "trade_ts_ns")
            if ts > end:
                break
            px = float(trade_prices[idx])
            idx += 1
            if not (isfinite(px) and px > 0):
                continue
            future_count += 1
            decision_dr = directional_bps(px, dp, direction)
            max_decision_dr = max(max_decision_dr, decision_dr)
            min_decision_dr = min(min_decision_dr, decision_dr)
            last_ts = ts
            last_px = px
            origin_dr = directional_bps(px, op, direction)
            for target in target_ids:
                if target_hit[target] is None and origin_dr + 1e-12 >= target:
                    target_hit[target] = ts

        row: dict[str, object] = {
            "outcome_end_ts_ns": end,
            "fully_observed": True,
            "future_trade_count": future_count,
            "mfe_from_decision_bps": max(0.0, max_decision_dr),
            "mae_from_decision_bps": max(0.0, -min_decision_dr),
            "directional_return_to_last_trade_bps": (
                directional_bps(last_px, dp, direction) if future_count else 0.0
            ),
            "last_trade_ts_ns": last_ts,
            "last_trade_price": last_px,
        }
        for target in target_ids:
            hit_ts = target_hit[target]
            row[f"origin_reach_{target}"] = bool(hit_ts is not None)
            row[f"origin_hit_{target}_ts_ns"] = hit_ts
            row[f"seconds_to_origin_{target}_from_outcome_start"] = (
                (hit_ts - start) / NS_PER_SECOND if hit_ts is not None else None
            )
        out[horizon] = row

    return out

def summarize_outcome_horizon(
    *,
    trade_ts_ns: Sequence[int],
    trade_prices: Sequence[float],
    outcome_start_ts_ns: int,
    decision_price: float,
    origin_price: float,
    direction: int,
    horizon_seconds: int,
    fully_observed: bool,
    targets_bps: Sequence[int] = OUTCOME_TARGETS_BPS,
) -> dict[str, object]:
    """Summarize one exact-anchor future horizon without pre-start leakage.

    Only trades strictly after ``outcome_start_ts_ns`` enter the future path.
    The decision price itself is treated as t=0 for return and target geometry.
    Labels are emitted only when ``fully_observed`` is true; otherwise target
    labels and return summaries remain ``None`` to avoid outcome-dependent
    attrition or partial-window bias.
    """

    if len(trade_ts_ns) != len(trade_prices):
        raise ValueError("trade timestamp and price arrays must have equal length")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be > 0")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    start = require_int64_ns(outcome_start_ts_ns, "outcome_start_ts_ns")
    end = start + int(horizon_seconds) * NS_PER_SECOND
    dp = float(decision_price)
    op = float(origin_price)
    if not (isfinite(dp) and dp > 0 and isfinite(op) and op > 0):
        raise ValueError("decision_price and origin_price must be finite and > 0")

    # bisect_right excludes a trade stamped exactly at the outcome start; the
    # decision price is the causal t=0 state and future rows begin strictly after.
    lo = bisect_right(trade_ts_ns, start)
    hi = bisect_right(trade_ts_ns, end)
    fut_ts = [int(x) for x in trade_ts_ns[lo:hi]]
    fut_px = [float(x) for x in trade_prices[lo:hi]]

    row: dict[str, object] = {
        "outcome_end_ts_ns": end,
        "fully_observed": bool(fully_observed),
        "future_trade_count": int(len(fut_ts)),
    }
    if not fully_observed:
        row.update(
            {
                "mfe_from_decision_bps": None,
                "mae_from_decision_bps": None,
                "directional_return_to_last_trade_bps": None,
                "last_trade_ts_ns": None,
                "last_trade_price": None,
            }
        )
        for target in targets_bps:
            row[f"origin_reach_{int(target)}"] = None
            row[f"origin_hit_{int(target)}_ts_ns"] = None
            row[f"seconds_to_origin_{int(target)}_from_outcome_start"] = None
        return row

    path_px = [dp] + fut_px
    decision_dr = [directional_bps(px, dp, direction) for px in path_px]
    row["mfe_from_decision_bps"] = max(0.0, max(decision_dr))
    row["mae_from_decision_bps"] = max(0.0, -min(decision_dr))
    if fut_ts:
        row["last_trade_ts_ns"] = fut_ts[-1]
        row["last_trade_price"] = fut_px[-1]
        row["directional_return_to_last_trade_bps"] = directional_bps(fut_px[-1], dp, direction)
    else:
        row["last_trade_ts_ns"] = start
        row["last_trade_price"] = dp
        row["directional_return_to_last_trade_bps"] = 0.0

    start_origin_dr = directional_bps(dp, op, direction)
    for target in targets_bps:
        target_i = int(target)
        hit_ts: int | None = start if start_origin_dr + 1e-12 >= target_i else None
        if hit_ts is None:
            for ts, px in zip(fut_ts, fut_px):
                if directional_bps(px, op, direction) + 1e-12 >= target_i:
                    hit_ts = int(ts)
                    break
        row[f"origin_reach_{target_i}"] = bool(hit_ts is not None)
        row[f"origin_hit_{target_i}_ts_ns"] = hit_ts
        row[f"seconds_to_origin_{target_i}_from_outcome_start"] = (
            (hit_ts - start) / NS_PER_SECOND if hit_ts is not None else None
        )
    return row
