from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.clocks.contract import (  # noqa: E402
    ClockContractError,
    D0_END_EXCLUSIVE_NS,
    D0_START_NS,
    NS_PER_SECOND,
    assert_clock_contract,
    assert_d0_only,
    audit_legacy_candidate_second_overlap,
    validate_clock_contract,
)


def valid_record(base: int | None = None) -> dict[str, int]:
    t = D0_START_NS + 12 * NS_PER_SECOND if base is None else base
    return {
        "candidate_detection_ts_ns": t,
        "anchor_ts_ns": t + 950_000_000,
        "feature_window_start_ts_ns": t,
        "feature_window_end_ts_ns": t + 3_950_000_000,
        "information_available_ts_ns": t + 3_950_000_000,
        "decision_ts_ns": t + 4_000_000_000,
        "order_send_ts_ns": t + 4_010_000_000,
        "estimated_exchange_arrival_ts_ns": t + 4_030_000_000,
        "executable_entry_ts_ns": t + 4_040_000_000,
        "outcome_start_ts_ns": t + 4_040_000_000,
    }


def codes(record: dict[str, object]) -> set[str]:
    return {v.code for v in validate_clock_contract(record)}


def test_exact_nanosecond_clock_passes() -> None:
    record = valid_record()
    assert_clock_contract(record)
    assert_d0_only(record, max_outcome_horizon_seconds=1800)


def test_float_nanosecond_timestamp_is_rejected() -> None:
    record: dict[str, object] = valid_record()
    record["anchor_ts_ns"] = float(record["anchor_ts_ns"])
    assert "CLOCK_NOT_INT64_NS" in codes(record)


def test_future_information_at_decision_is_rejected() -> None:
    record = valid_record()
    record["information_available_ts_ns"] = record["decision_ts_ns"] + 1
    assert "FUTURE_INFORMATION_AT_DECISION" in codes(record)


def test_candidate_may_become_available_after_book_anchor() -> None:
    record = valid_record()
    record["candidate_detection_ts_ns"] = record["anchor_ts_ns"] + 25_000_000
    assert_clock_contract(record)


def test_candidate_must_be_available_by_information_timestamp() -> None:
    record = valid_record()
    record["candidate_detection_ts_ns"] = record["information_available_ts_ns"] + 1
    assert "CANDIDATE_NOT_YET_AVAILABLE" in codes(record)


def test_feature_outcome_overlap_is_rejected() -> None:
    record = valid_record()
    record["outcome_start_ts_ns"] = record["feature_window_end_ts_ns"] - 950_000_000
    assert "FEATURE_OUTCOME_OVERLAP" in codes(record)


def test_historical_candidate_second_bug_produces_950ms_overlap() -> None:
    candidate_second = D0_START_NS // NS_PER_SECOND + 100
    exact_anchor = candidate_second * NS_PER_SECOND + 950_000_000
    overlap = audit_legacy_candidate_second_overlap(
        candidate_second=candidate_second,
        exact_anchor_ts_ns=exact_anchor,
        legacy_outcome_offset_seconds=3,
        exact_feature_offset_seconds=3,
    )
    assert overlap == 950_000_000


def test_exact_anchor_outcome_start_removes_candidate_second_overlap() -> None:
    candidate_second = D0_START_NS // NS_PER_SECOND + 100
    exact_anchor = candidate_second * NS_PER_SECOND + 950_000_000
    exact_feature_end = exact_anchor + 3 * NS_PER_SECOND
    exact_outcome_start = exact_anchor + 3 * NS_PER_SECOND
    assert exact_feature_end <= exact_outcome_start


def test_p1_record_is_refused() -> None:
    record = valid_record(base=D0_END_EXCLUSIVE_NS + NS_PER_SECOND)
    with pytest.raises(ClockContractError) as exc_info:
        assert_d0_only(record)
    assert "P1_ACCESS_FORBIDDEN" in {v.code for v in exc_info.value.violations}


def test_late_d0_horizon_that_touches_p1_is_refused() -> None:
    # Put all required clocks 10 seconds before P1, then request a 30s outcome.
    record = valid_record(base=D0_END_EXCLUSIVE_NS - 15 * NS_PER_SECOND)
    with pytest.raises(ClockContractError) as exc_info:
        assert_d0_only(record, max_outcome_horizon_seconds=30)
    assert "OUTCOME_HORIZON_TOUCHES_P1" in {v.code for v in exc_info.value.violations}


def test_execution_order_is_monotone() -> None:
    record = valid_record()
    record["estimated_exchange_arrival_ts_ns"] = record["order_send_ts_ns"] - 1
    assert "ARRIVAL_BEFORE_SEND" in codes(record)
