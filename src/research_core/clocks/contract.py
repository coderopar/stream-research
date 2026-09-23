"""R0.2 causal-clock contract.

All governed timestamps are signed int64 Unix nanoseconds.  Floating-point
clock values are rejected deliberately: a float cannot safely represent modern
nanosecond epoch timestamps exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from typing import Mapping, Sequence

NS_PER_SECOND = 1_000_000_000
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def _utc_ns(year: int, month: int, day: int) -> int:
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return int(dt.timestamp()) * NS_PER_SECOND


D0_START_NS = _utc_ns(2026, 5, 18)
D0_END_EXCLUSIVE_NS = _utc_ns(2026, 7, 19)

REQUIRED_CLOCK_FIELDS: tuple[str, ...] = (
    "candidate_detection_ts_ns",
    "anchor_ts_ns",
    "feature_window_start_ts_ns",
    "feature_window_end_ts_ns",
    "information_available_ts_ns",
    "decision_ts_ns",
    "order_send_ts_ns",
    "estimated_exchange_arrival_ts_ns",
    "executable_entry_ts_ns",
    "outcome_start_ts_ns",
)


@dataclass(frozen=True)
class ClockViolation:
    code: str
    message: str
    fields: tuple[str, ...] = ()


class ClockContractError(ValueError):
    """Raised when a record violates the governed causal-clock contract."""

    def __init__(self, violations: Sequence[ClockViolation]):
        self.violations = tuple(violations)
        detail = "; ".join(f"{v.code}: {v.message}" for v in self.violations)
        super().__init__(detail)


def require_int64_ns(value: object, field: str) -> int:
    """Return an integer nanosecond timestamp or raise a typed contract error.

    bool is rejected even though it subclasses int. Floats are rejected rather
    than rounded/coerced because that can destroy sub-microsecond information.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ClockContractError(
            [
                ClockViolation(
                    "CLOCK_NOT_INT64_NS",
                    f"{field} must be an integer Unix-nanosecond timestamp; got {type(value).__name__}",
                    (field,),
                )
            ]
        )
    ivalue = int(value)
    if ivalue < INT64_MIN or ivalue > INT64_MAX:
        raise ClockContractError(
            [
                ClockViolation(
                    "CLOCK_OUTSIDE_INT64",
                    f"{field}={ivalue} is outside signed int64 range",
                    (field,),
                )
            ]
        )
    return ivalue


def _read_required(record: Mapping[str, object]) -> tuple[dict[str, int], list[ClockViolation]]:
    values: dict[str, int] = {}
    violations: list[ClockViolation] = []
    for field in REQUIRED_CLOCK_FIELDS:
        if field not in record:
            violations.append(
                ClockViolation("CLOCK_FIELD_MISSING", f"required field {field} is missing", (field,))
            )
            continue
        try:
            values[field] = require_int64_ns(record[field], field)
        except ClockContractError as exc:
            violations.extend(exc.violations)
    return values, violations


def validate_clock_contract(record: Mapping[str, object]) -> tuple[ClockViolation, ...]:
    """Return every clock violation found in one governed record."""

    ts, violations = _read_required(record)
    if violations:
        return tuple(violations)

    def le(left: str, right: str, code: str, reason: str) -> None:
        if ts[left] > ts[right]:
            violations.append(ClockViolation(code, reason, (left, right)))

    # Candidate availability and the recovered book-state anchor are distinct
    # clocks. For the frozen one-second candidate policy, the candidate can
    # become available after the exact depth state that underlies the end-of-
    # second book. Their mutual ordering is therefore intentionally unconstrained.
    # Both must, however, be known by the record's information-availability time.
    le(
        "candidate_detection_ts_ns",
        "information_available_ts_ns",
        "CANDIDATE_NOT_YET_AVAILABLE",
        "candidate_detection_ts_ns must be <= information_available_ts_ns",
    )
    le(
        "feature_window_start_ts_ns",
        "feature_window_end_ts_ns",
        "FEATURE_WINDOW_REVERSED",
        "feature window start cannot occur after feature window end",
    )
    le(
        "feature_window_end_ts_ns",
        "information_available_ts_ns",
        "FEATURE_AFTER_INFORMATION_AVAILABILITY",
        "a feature window cannot end after the information-availability timestamp assigned to it",
    )
    le(
        "anchor_ts_ns",
        "information_available_ts_ns",
        "ANCHOR_NOT_YET_AVAILABLE",
        "the exact anchor must be observed before the record is declared information-available",
    )
    le(
        "information_available_ts_ns",
        "decision_ts_ns",
        "FUTURE_INFORMATION_AT_DECISION",
        "information_available_ts_ns must be <= decision_ts_ns",
    )
    le(
        "decision_ts_ns",
        "order_send_ts_ns",
        "ORDER_SENT_BEFORE_DECISION",
        "order_send_ts_ns must be >= decision_ts_ns",
    )
    le(
        "order_send_ts_ns",
        "estimated_exchange_arrival_ts_ns",
        "ARRIVAL_BEFORE_SEND",
        "estimated exchange arrival cannot precede order send",
    )
    le(
        "estimated_exchange_arrival_ts_ns",
        "executable_entry_ts_ns",
        "ENTRY_BEFORE_ARRIVAL",
        "executable entry cannot precede estimated exchange arrival",
    )
    le(
        "executable_entry_ts_ns",
        "outcome_start_ts_ns",
        "OUTCOME_BEFORE_ENTRY",
        "outcome_start_ts_ns must be >= executable_entry_ts_ns",
    )

    if ts["feature_window_end_ts_ns"] > ts["outcome_start_ts_ns"]:
        overlap = ts["feature_window_end_ts_ns"] - ts["outcome_start_ts_ns"]
        violations.append(
            ClockViolation(
                "FEATURE_OUTCOME_OVERLAP",
                f"feature window overlaps the governed outcome interval by {overlap} ns",
                ("feature_window_end_ts_ns", "outcome_start_ts_ns"),
            )
        )

    return tuple(violations)


def assert_clock_contract(record: Mapping[str, object]) -> None:
    violations = validate_clock_contract(record)
    if violations:
        raise ClockContractError(violations)


def assert_d0_only(
    record: Mapping[str, object],
    *,
    max_outcome_horizon_seconds: int = 0,
) -> None:
    """Refuse any governed record that would touch P1_PROTECTED.

    The maximum outcome horizon is included because a late-D0 candidate could
    have an outcome_start inside D0 while a 15m/30m label reaches across the P1
    boundary. R0.2 must never inspect such a label.
    """

    if max_outcome_horizon_seconds < 0:
        raise ValueError("max_outcome_horizon_seconds must be >= 0")

    values, violations = _read_required(record)
    if violations:
        raise ClockContractError(violations)

    boundary_violations: list[ClockViolation] = []
    for field, value in values.items():
        if value < D0_START_NS:
            boundary_violations.append(
                ClockViolation(
                    "BEFORE_D0",
                    f"{field} is before the frozen D0 start",
                    (field,),
                )
            )
        if value >= D0_END_EXCLUSIVE_NS:
            boundary_violations.append(
                ClockViolation(
                    "P1_ACCESS_FORBIDDEN",
                    f"{field} reaches P1_PROTECTED or later",
                    (field,),
                )
            )

    outcome_end = values["outcome_start_ts_ns"] + max_outcome_horizon_seconds * NS_PER_SECOND
    if outcome_end >= D0_END_EXCLUSIVE_NS:
        boundary_violations.append(
            ClockViolation(
                "OUTCOME_HORIZON_TOUCHES_P1",
                "the requested D0 outcome horizon reaches P1_PROTECTED; exclude this candidate",
                ("outcome_start_ts_ns",),
            )
        )

    if boundary_violations:
        raise ClockContractError(boundary_violations)


def audit_legacy_candidate_second_overlap(
    *,
    candidate_second: int,
    exact_anchor_ts_ns: int,
    legacy_outcome_offset_seconds: int,
    exact_feature_offset_seconds: int,
) -> int:
    """Return legacy feature/outcome overlap in nanoseconds.

    This is a diagnostic only. It intentionally does *not* construct a repaired
    timestamp from candidate_second. A positive result demonstrates that the
    integer-second outcome start occurs before the exact-anchor feature window
    has finished.
    """

    if isinstance(candidate_second, bool) or not isinstance(candidate_second, Integral):
        raise TypeError("candidate_second must be an integer epoch second")
    anchor = require_int64_ns(exact_anchor_ts_ns, "exact_anchor_ts_ns")
    if legacy_outcome_offset_seconds < 0 or exact_feature_offset_seconds < 0:
        raise ValueError("offsets must be >= 0")

    legacy_outcome_start_ns = (int(candidate_second) + legacy_outcome_offset_seconds) * NS_PER_SECOND
    exact_feature_end_ns = anchor + exact_feature_offset_seconds * NS_PER_SECOND
    return max(0, exact_feature_end_ns - legacy_outcome_start_ns)
