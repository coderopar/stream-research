from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.clocks.contract import ClockContractError, NS_PER_SECOND  # noqa: E402
from research_core.substrate.d0 import (  # noqa: E402
    assert_causal_columns_future_clean,
    build_scientific_label_clock,
    earliest_threshold_crossing_in_second,
    frozen_candidate_availability_ts_ns,
    latest_prior_trade,
    legacy_timing_order_valid,
    summarize_outcome_horizon,
    summarize_outcome_horizons,
    validate_exact_anchor,
)


def test_exact_anchor_inside_legacy_second_is_accepted() -> None:
    csec = 1_779_235_200
    anchor = csec * NS_PER_SECOND + 950_000_000
    assert validate_exact_anchor(csec, anchor) == anchor


def test_anchor_is_not_derived_from_candidate_second() -> None:
    csec = 1_779_235_200
    with pytest.raises(ClockContractError):
        validate_exact_anchor(csec, float(csec * NS_PER_SECOND + 950_000_000))


def test_anchor_outside_candidate_second_is_rejected() -> None:
    csec = 1_779_235_200
    with pytest.raises(ValueError):
        validate_exact_anchor(csec, (csec + 2) * NS_PER_SECOND)




def test_anchor_exactly_at_next_second_is_rejected() -> None:
    csec = 1_779_235_200
    with pytest.raises(ValueError):
        validate_exact_anchor(csec, (csec + 1) * NS_PER_SECOND)


def test_scientific_clock_starts_outcome_at_anchor_plus_3s() -> None:
    anchor = 1_779_235_200_950_000_000
    candidate_available = 1_779_235_201_000_000_000
    clock = build_scientific_label_clock(
        anchor, candidate_detection_ts_ns=candidate_available
    )
    expected = anchor + 3 * NS_PER_SECOND
    assert clock["candidate_detection_ts_ns"] == candidate_available
    assert clock["candidate_detection_ts_ns"] > clock["anchor_ts_ns"]
    assert clock["feature_window_end_ts_ns"] == expected
    assert clock["outcome_start_ts_ns"] == expected


def test_candidate_may_become_available_after_anchor_but_before_decision() -> None:
    anchor = 1_779_235_200_950_000_000
    clock = build_scientific_label_clock(
        anchor, candidate_detection_ts_ns=anchor + 50_000_000
    )
    assert clock["candidate_detection_ts_ns"] > clock["anchor_ts_ns"]


def test_candidate_cannot_become_available_after_decision() -> None:
    anchor = 1_779_235_200_950_000_000
    decision = anchor + 3 * NS_PER_SECOND
    with pytest.raises(ValueError):
        build_scientific_label_clock(
            anchor, candidate_detection_ts_ns=decision + 1
        )

def test_observation_horizon_is_frozen_at_three_seconds() -> None:
    with pytest.raises(ValueError):
        build_scientific_label_clock(1_779_235_200_950_000_000, observation_horizon_seconds=5)


def test_causal_schema_rejects_future_like_columns() -> None:
    with pytest.raises(ValueError):
        assert_causal_columns_future_clean(["candidate_id", "h900_origin_reach_75"])


def test_frozen_candidate_availability_is_interval_close() -> None:
    assert frozen_candidate_availability_ts_ns(100) == 101 * NS_PER_SECOND


def test_legacy_timing_allows_same_second_release_and_candidate() -> None:
    assert legacy_timing_order_valid(100, 101, 102) is True
    assert legacy_timing_order_valid(100, 102, 102) is True
    assert legacy_timing_order_valid(102, 102, 102) is True
    assert legacy_timing_order_valid(103, 102, 102) is False


def test_exact_detection_is_half_open_and_preserves_equal_timestamp_hits() -> None:
    csec = 100
    start = csec * NS_PER_SECOND
    ts = [start - 1, start + 100, start + 100, start + 500, start + NS_PER_SECOND]
    px = [101.0, 100.1, 100.3, 100.4, 105.0]
    hit = earliest_threshold_crossing_in_second(
        trade_ts_ns=ts,
        trade_prices=px,
        candidate_second=csec,
        origin_price=100.0,
        direction=1,
        threshold_bps=25.0,
    )
    assert hit == start + 100


def test_latest_prior_trade_never_selects_future_trade() -> None:
    point = 10 * NS_PER_SECOND
    ts = [point - 100_000_000, point + 1]
    px = [100.0, 999.0]
    got_ts, got_px = latest_prior_trade(ts, px, point)
    assert got_ts == ts[0]
    assert got_px == 100.0


def test_latest_prior_trade_respects_staleness() -> None:
    point = 10 * NS_PER_SECOND
    ts = [point - 6 * NS_PER_SECOND]
    px = [100.0]
    got_ts, got_px = latest_prior_trade(ts, px, point)
    assert got_ts is None and got_px is None


def test_outcome_path_excludes_pre_start_trade() -> None:
    start = 10 * NS_PER_SECOND
    # The pre-start 1% spike would hit 50bp if leakage were present.  The only
    # post-start trade is flat, so the target must remain false.
    ts = [start - 1, start + 1]
    px = [101.0, 100.0]
    out = summarize_outcome_horizon(
        trade_ts_ns=ts,
        trade_prices=px,
        outcome_start_ts_ns=start,
        decision_price=100.0,
        origin_price=100.0,
        direction=1,
        horizon_seconds=300,
        fully_observed=True,
        targets_bps=(50,),
    )
    assert out["origin_reach_50"] is False


def test_outcome_first_passage_uses_exact_ns() -> None:
    start = 10 * NS_PER_SECOND + 950_000_000
    ts = [start + 10_000_000, start + 25_000_000]
    px = [100.2, 100.8]
    out = summarize_outcome_horizon(
        trade_ts_ns=ts,
        trade_prices=px,
        outcome_start_ts_ns=start,
        decision_price=100.0,
        origin_price=100.0,
        direction=1,
        horizon_seconds=300,
        fully_observed=True,
        targets_bps=(50,),
    )
    assert out["origin_reach_50"] is True
    assert out["origin_hit_50_ts_ns"] == ts[1]
    assert out["seconds_to_origin_50_from_outcome_start"] == pytest.approx(0.025)


def test_incomplete_window_emits_no_outcome_values() -> None:
    start = 10 * NS_PER_SECOND
    out = summarize_outcome_horizon(
        trade_ts_ns=[start + 1],
        trade_prices=[110.0],
        outcome_start_ts_ns=start,
        decision_price=100.0,
        origin_price=100.0,
        direction=1,
        horizon_seconds=300,
        fully_observed=False,
        targets_bps=(50,),
    )
    assert out["origin_reach_50"] is None
    assert out["mfe_from_decision_bps"] is None


def _load_r03_builder_module():
    import importlib.util

    pytest.importorskip("pyarrow")
    path = REPO_ROOT / "research" / "r0" / "substrate" / "build_corrected_d0_substrate.py"
    spec = importlib.util.spec_from_file_location("r0_3_builder_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_refuses_non_d0_source_dates(tmp_path: Path) -> None:
    b = _load_r03_builder_module()
    with pytest.raises(RuntimeError, match="refuses non-D0 canonical date"):
        b.canonical_trade_paths(tmp_path, "2026-07-19")
    with pytest.raises(RuntimeError, match="refuses non-D0 execution date"):
        b.execution_day_path(tmp_path, "2026-07-19")


def test_trade_loader_preserves_equal_receive_timestamps(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    b = _load_r03_builder_module()
    day = "2026-06-01"
    root = tmp_path / "canonical"
    part = root / "perps" / "trades" / f"date={day}" / "part-0.parquet"
    part.parent.mkdir(parents=True)
    table = pa.table({
        "ts_recv_ns": pa.array([10, 10, 11], type=pa.int64()),
        "price": ["100.0", "100.3", "100.4"],
        "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "source_file_id": ["a", "a", "a"],
        "source_record_number": pa.array([1, 2, 3], type=pa.int64()),
        "aggregate_trade_id": pa.array([1, 2, 3], type=pa.int64()),
    })
    pq.write_table(table, part)
    ts, px, _ = b.load_day_trade_arrays(root, [day])
    assert ts == [10, 10, 11]
    assert px == [100.0, 100.3, 100.4]


def test_float_anchor_metrics_falls_back_to_int64_level_fate(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    b = _load_r03_builder_module()
    day = "2026-06-01"
    cid = "c1"
    root = tmp_path / "anchors"
    mp = root / "candidate_metrics" / f"candidate_date={day}" / "part.parquet"
    lp = root / "level_fate" / f"candidate_date={day}" / "part.parquet"
    mp.parent.mkdir(parents=True)
    lp.parent.mkdir(parents=True)
    # Historical nullable pandas materialization can make this float64; R0.3
    # must not cast it back to an exact epoch.
    pq.write_table(pa.table({
        "candidate_id": [cid],
        "observation_horizon_sec": [3],
        "anchor_found": [True],
        "sequence_valid": [True],
        "invalid_reason": [None],
        "anchor_ts_recv_ns": pa.array([1.75e18], type=pa.float64()),
    }), mp)
    exact_anchor = 1_749_000_000_950_000_000
    pq.write_table(pa.table({
        "candidate_id": [cid],
        "anchor_ts_recv_ns": pa.array([exact_anchor], type=pa.int64()),
        "anchor_found": [True],
    }), lp)
    candidates = pd.DataFrame({"candidate_id": [cid], "candidate_date": [day]})
    exact, reasons, _ = b.load_exact_anchor_index(root, candidates)
    assert exact[cid] == exact_anchor
    assert cid not in reasons


def test_multi_horizon_summary_matches_single_horizon_semantics() -> None:
    start = 10 * NS_PER_SECOND + 950_000_000
    ts = [
        start + 1,
        start + 100 * NS_PER_SECOND,
        start + 300 * NS_PER_SECOND,
        start + 301 * NS_PER_SECOND,
        start + 600 * NS_PER_SECOND,
    ]
    px = [100.0, 100.4, 100.8, 99.5, 101.2]
    horizons = (300, 600)
    multi = summarize_outcome_horizons(
        trade_ts_ns=ts,
        trade_prices=px,
        outcome_start_ts_ns=start,
        decision_price=100.0,
        origin_price=100.0,
        direction=1,
        horizons_seconds=horizons,
        targets_bps=(50,),
    )
    for horizon in horizons:
        single = summarize_outcome_horizon(
            trade_ts_ns=ts,
            trade_prices=px,
            outcome_start_ts_ns=start,
            decision_price=100.0,
            origin_price=100.0,
            direction=1,
            horizon_seconds=horizon,
            fully_observed=True,
            targets_bps=(50,),
        )
        assert multi[horizon] == single


def test_multi_horizon_includes_trade_exactly_at_horizon_end() -> None:
    start = 10 * NS_PER_SECOND
    end_300 = start + 300 * NS_PER_SECOND
    multi = summarize_outcome_horizons(
        trade_ts_ns=[end_300, end_300 + 1],
        trade_prices=[100.8, 101.5],
        outcome_start_ts_ns=start,
        decision_price=100.0,
        origin_price=100.0,
        direction=1,
        horizons_seconds=(300,),
        targets_bps=(50,),
    )
    assert multi[300]["future_trade_count"] == 1
    assert multi[300]["origin_reach_50"] is True
    assert multi[300]["origin_hit_50_ts_ns"] == end_300
