from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_core.dependence import (
    assert_component_refinement,
    connected_interval_components,
    conservative_day_overlap_clusters,
    exact_lineage_groups,
    gap_connected_episodes,
    partition_episodes_by_value,
)


def test_half_open_touching_boundaries_do_not_overlap():
    a = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 10},
            {"candidate_id": "b", "start_ns": 10, "end_ns": 20},
        ],
        prefix="T",
    )
    assert a["a"].component_id != a["b"].component_id


def test_true_overlap_is_connected():
    a = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 10},
            {"candidate_id": "b", "start_ns": 9, "end_ns": 20},
        ],
        prefix="T",
    )
    assert a["a"].component_id == a["b"].component_id
    assert a["a"].component_size == 2


def test_overlap_components_are_transitive():
    a = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 10},
            {"candidate_id": "b", "start_ns": 9, "end_ns": 20},
            {"candidate_id": "c", "start_ns": 19, "end_ns": 30},
        ],
        prefix="T",
    )
    assert len({a[k].component_id for k in "abc"}) == 1


def test_overlap_ids_are_stable_to_input_order():
    rows = [
        {"candidate_id": "b", "start_ns": 9, "end_ns": 20},
        {"candidate_id": "a", "start_ns": 0, "end_ns": 10},
        {"candidate_id": "c", "start_ns": 30, "end_ns": 40},
    ]
    forward = connected_interval_components(rows, prefix="T")
    reverse = connected_interval_components(list(reversed(rows)), prefix="T")
    assert {k: v.component_id for k, v in forward.items()} == {
        k: v.component_id for k, v in reverse.items()
    }


def test_duplicate_interval_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate interval id"):
        connected_interval_components(
            [
                {"candidate_id": "a", "start_ns": 0, "end_ns": 10},
                {"candidate_id": "a", "start_ns": 20, "end_ns": 30},
            ]
        )


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError, match="end > start"):
        connected_interval_components(
            [{"candidate_id": "a", "start_ns": 10, "end_ns": 10}]
        )


def test_nested_horizon_components_refine_coarser_components():
    fine = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 5},
            {"candidate_id": "b", "start_ns": 4, "end_ns": 9},
            {"candidate_id": "c", "start_ns": 20, "end_ns": 25},
        ],
        prefix="F",
    )
    coarse = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 30},
            {"candidate_id": "b", "start_ns": 4, "end_ns": 34},
            {"candidate_id": "c", "start_ns": 20, "end_ns": 50},
        ],
        prefix="C",
    )
    assert_component_refinement(fine, coarse)


def test_refinement_violation_is_detected():
    fine = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 10},
            {"candidate_id": "b", "start_ns": 9, "end_ns": 20},
        ],
        prefix="F",
    )
    coarse = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 5},
            {"candidate_id": "b", "start_ns": 10, "end_ns": 15},
        ],
        prefix="C",
    )
    with pytest.raises(AssertionError, match="maps to more than one"):
        assert_component_refinement(fine, coarse)


def test_gap_episode_includes_exact_threshold_and_breaks_above_it():
    ep = gap_connected_episodes(
        [
            {"candidate_id": "a", "candidate_time_ns": 0},
            {"candidate_id": "b", "candidate_time_ns": 60},
            {"candidate_id": "c", "candidate_time_ns": 121},
        ],
        max_gap_ns=60,
        prefix="E",
    )
    assert ep["a"].episode_id == ep["b"].episode_id
    assert ep["b"].episode_id != ep["c"].episode_id


def test_same_timestamp_candidates_share_temporal_episode():
    ep = gap_connected_episodes(
        [
            {"candidate_id": "a", "candidate_time_ns": 10},
            {"candidate_id": "b", "candidate_time_ns": 10},
        ],
        max_gap_ns=0,
        prefix="E",
    )
    assert ep["a"].episode_id == ep["b"].episode_id


def test_direction_partition_does_not_depend_on_simultaneous_ordering():
    ep = gap_connected_episodes(
        [
            {"candidate_id": "a", "candidate_time_ns": 10},
            {"candidate_id": "b", "candidate_time_ns": 10},
            {"candidate_id": "c", "candidate_time_ns": 20},
        ],
        max_gap_ns=60,
        prefix="E",
    )
    d = partition_episodes_by_value(ep, {"a": 1, "b": -1, "c": 1}, prefix="D")
    assert d["a"].episode_id == d["c"].episode_id
    assert d["a"].episode_id != d["b"].episode_id


def test_exact_origin_lineage_groups_exact_tuple_only():
    groups = exact_lineage_groups(
        [
            {"candidate_id": "a", "candidate_time_ns": 10, "direction": 1, "origin": 1},
            {"candidate_id": "b", "candidate_time_ns": 20, "direction": 1, "origin": 1},
            {"candidate_id": "c", "candidate_time_ns": 21, "direction": -1, "origin": 1},
        ],
        group_fields=("direction", "origin"),
        prefix="L",
    )
    assert groups["a"].episode_id == groups["b"].episode_id
    assert groups["a"].episode_id != groups["c"].episode_id


def test_conservative_cluster_groups_same_day_even_without_overlap():
    day = {"a": "2026-01-01", "b": "2026-01-01"}
    clusters = conservative_day_overlap_clusters(day, {}, prefix="X")
    assert clusters["a"] == clusters["b"]


def test_conservative_cluster_unions_days_when_30m_component_crosses_midnight():
    overlap = connected_interval_components(
        [
            {"candidate_id": "a", "start_ns": 0, "end_ns": 20},
            {"candidate_id": "b", "start_ns": 10, "end_ns": 30},
        ],
        prefix="O",
    )
    day = {"a": "2026-01-01", "b": "2026-01-02", "c": "2026-01-02"}
    clusters = conservative_day_overlap_clusters(day, overlap, prefix="X")
    assert clusters["a"] == clusters["b"] == clusters["c"]


def test_clock_unavailable_candidate_can_still_receive_temporal_episode():
    # R0.4 temporal episodes require the frozen candidate-availability clock,
    # not an exact book anchor / R0.3 governed label clock.
    ep = gap_connected_episodes(
        [{"candidate_id": "missing-anchor", "candidate_time_ns": 100}],
        max_gap_ns=60,
        prefix="E",
    )
    assert ep["missing-anchor"].episode_size == 1


def test_r04_implementation_declares_no_future_outcome_read():
    script = Path("research/r0/dependence/build_d0_dependence_framework.py").read_text(
        encoding="utf-8"
    )
    assert "future_outcome_values_read\": False" in script
    assert 'r0_3_future_outcomes_path_opened\": False' in script
    assert "future/outcomes.parquet" not in script


def test_dependence_contract_is_outcome_blind_and_frozen():
    contract = json.loads(
        Path("research/governance/dependence_contract_v1.json").read_text(encoding="utf-8")
    )
    assert contract["outcome_firewall"]["future_outcome_values_read"] is False
    assert contract["protected_data"]["p1_read"] is False
    assert contract["episodes"]["policy_temporal_gap_sec"] == 60
    assert contract["overlap"]["touching_boundaries_overlap"] is False


def _load_r04_builder_module():
    import importlib.util
    path = Path("research/r0/dependence/build_d0_dependence_framework.py")
    spec = importlib.util.spec_from_file_location("r04_builder_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_framework_preserves_clock_unavailable_candidate_and_gap_fields():
    pd = pytest.importorskip("pandas")
    m = _load_r04_builder_module()
    sec0 = m.D0_START_NS // m.NS_PER_SECOND
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "candidate_date": "2026-05-18", "direction": 1, "legacy_origin_second": sec0 + 100, "legacy_candidate_second": sec0 + 200, "cooldown_sec": 60, "clock_available": True, "clock_unavailable_reason": None},
            {"candidate_id": "b", "candidate_date": "2026-05-18", "direction": -1, "legacy_origin_second": sec0 + 110, "legacy_candidate_second": sec0 + 230, "cooldown_sec": 60, "clock_available": True, "clock_unavailable_reason": None},
            {"candidate_id": "c", "candidate_date": "2026-05-18", "direction": 1, "legacy_origin_second": sec0 + 100, "legacy_candidate_second": sec0 + 500, "cooldown_sec": 60, "clock_available": False, "clock_unavailable_reason": "MISSING"},
        ]
    )
    clocks = pd.DataFrame(
        [
            {"candidate_id": "a", "outcome_start_ts_ns": m.D0_START_NS + 203 * m.NS_PER_SECOND},
            {"candidate_id": "b", "outcome_start_ts_ns": m.D0_START_NS + 233 * m.NS_PER_SECOND},
        ]
    )
    frame, _, _, _, summary = m.build_framework(candidates, clocks)
    by_id = frame.set_index("candidate_id")
    assert len(frame) == 3
    assert pd.isna(by_id.loc["c", "overlap_component_5m"])
    assert by_id.loc["a", "next_candidate_gap_sec"] == 30.0
    assert by_id.loc["b", "prev_candidate_gap_sec"] == 30.0
    assert summary["clock_unavailable_rows"] == 1


def test_build_framework_horizon_components_are_nested():
    pd = pytest.importorskip("pandas")
    m = _load_r04_builder_module()
    sec0 = m.D0_START_NS // m.NS_PER_SECOND
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "candidate_date": "2026-05-18", "direction": 1, "legacy_origin_second": sec0 + 1, "legacy_candidate_second": sec0 + 10, "cooldown_sec": 60, "clock_available": True, "clock_unavailable_reason": None},
            {"candidate_id": "b", "candidate_date": "2026-05-18", "direction": 1, "legacy_origin_second": sec0 + 2, "legacy_candidate_second": sec0 + 400, "cooldown_sec": 60, "clock_available": True, "clock_unavailable_reason": None},
        ]
    )
    clocks = pd.DataFrame(
        [
            {"candidate_id": "a", "outcome_start_ts_ns": m.D0_START_NS + 20 * m.NS_PER_SECOND},
            {"candidate_id": "b", "outcome_start_ts_ns": m.D0_START_NS + 410 * m.NS_PER_SECOND},
        ]
    )
    frame, _, _, _, _ = m.build_framework(candidates, clocks)
    by_id = frame.set_index("candidate_id")
    # 5m windows are separate (390s start gap > 300s), while 10m+ overlap.
    assert by_id.loc["a", "overlap_component_5m"] != by_id.loc["b", "overlap_component_5m"]
    assert by_id.loc["a", "overlap_component_10m"] == by_id.loc["b", "overlap_component_10m"]
