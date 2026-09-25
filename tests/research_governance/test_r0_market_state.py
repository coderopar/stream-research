from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.clocks.contract import ClockContractError, D0_START_NS, NS_PER_SECOND  # noqa: E402
from research_core.market_state import (  # noqa: E402
    assert_available_by_cutoff,
    assert_feature_columns_safe,
    latest_prior_index,
    relative_bucket_bounds,
    summarize_trade_bucket,
)


def _builder():
    p = REPO_ROOT / "research" / "r0" / "market_state" / "build_d0_market_state_store.py"
    spec = importlib.util.spec_from_file_location("r05_builder", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_asof_selects_record_exactly_at_cutoff():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    got = latest_prior_index([cutoff - 1, cutoff, cutoff + 1], cutoff, deterministic_tie_keys=[1, 2, 3])
    assert got is not None
    assert got.index == 1
    assert got.ts_ns == cutoff
    assert got.age_ns == 0


def test_asof_never_selects_future_record():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    got = latest_prior_index([cutoff - 5, cutoff + 1], cutoff, deterministic_tie_keys=[1, 2])
    assert got is not None and got.ts_ns == cutoff - 5


def test_asof_rejects_float_epoch_ns():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    with pytest.raises(ClockContractError):
        latest_prior_index([float(cutoff)], cutoff)


def test_equal_timestamp_without_tie_key_is_explicitly_ambiguous():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    got = latest_prior_index([cutoff, cutoff], cutoff)
    assert got is not None and got.tie_ambiguous is True


def test_equal_timestamp_with_tie_keys_is_not_ambiguous():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    got = latest_prior_index([cutoff, cutoff], cutoff, deterministic_tie_keys=[("a", 1), ("a", 2)])
    assert got is not None and got.tie_ambiguous is False and got.index == 1


def test_available_by_cutoff_rejects_future_source():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND
    with pytest.raises(ValueError, match="after causal cutoff"):
        assert_available_by_cutoff(cutoff + 1, cutoff)


def test_relative_buckets_end_exactly_at_cutoff():
    cutoff = D0_START_NS + 10 * NS_PER_SECOND + 123
    b = relative_bucket_bounds(cutoff, lookback_sec=3, bucket_sec=1)
    assert [x[0] for x in b] == [-3, -2, -1]
    assert b[-1][2] == cutoff
    assert all(e - s == NS_PER_SECOND for _, s, e in b)


def test_relative_bucket_grid_requires_exact_divisibility():
    with pytest.raises(ValueError):
        relative_bucket_bounds(D0_START_NS + 10 * NS_PER_SECOND, lookback_sec=5, bucket_sec=2)


def test_trade_bucket_is_half_open_and_excludes_cutoff():
    start = D0_START_NS + 10 * NS_PER_SECOND
    end = start + NS_PER_SECOND
    b = summarize_trade_bucket(
        trade_ts_ns=[start, end - 1, end],
        trade_prices=[100, 101, 999],
        trade_quantities=[1, 2, 100],
        buyer_is_maker=[True, False, True],
        start_ts_ns=start,
        end_ts_ns=end,
    )
    assert b.trade_count == 2
    assert b.quantity_sum == 3
    assert b.notional_sum == 302
    assert b.buyer_maker_true_quantity == 1
    assert b.buyer_maker_false_quantity == 2
    assert b.last_trade_ts_ns == end - 1


def test_trade_bucket_preserves_equal_timestamp_rows():
    start = D0_START_NS + 10 * NS_PER_SECOND
    b = summarize_trade_bucket(
        trade_ts_ns=[start + 1, start + 1],
        trade_prices=[100, 101],
        trade_quantities=[1, 2],
        buyer_is_maker=[True, False],
        start_ts_ns=start,
        end_ts_ns=start + NS_PER_SECOND,
    )
    assert b.trade_count == 2 and b.quantity_sum == 3


def test_feature_guard_rejects_future_columns():
    with pytest.raises(ValueError, match="future"):
        assert_feature_columns_safe(["candidate_id", "p75_reached"])


def test_feature_guard_rejects_governance_columns():
    with pytest.raises(ValueError, match="governance"):
        assert_feature_columns_safe(["trade_count", "overlap_component_15m"])


def test_builder_refuses_p1_trade_date():
    m = _builder()
    with pytest.raises(RuntimeError, match="non-D0"):
        m.canonical_trade_paths(Path("x"), "2026-07-19")


def test_interval_partition_status_marks_pre_d0_history_unavailable():
    m = _builder()
    assert m.interval_partition_status(D0_START_NS - 1, D0_START_NS + 1, {"2026-05-18"}) == "OUTSIDE_D0_HISTORY"


def test_bucket_prefix_preserves_zero_observed_trades():
    pytest.importorskip("numpy")
    m = _builder()
    arr = m.prepare_trade_arrays(m.pd.DataFrame(columns=["ts_recv_ns", "price", "quantity", "buyer_is_maker"]))
    got = m.bucket_from_prefix(arr, D0_START_NS, D0_START_NS + NS_PER_SECOND)
    assert got["trade_count"] == 0 and got["quantity_sum"] == 0.0


def test_governance_linkage_does_not_copy_future_window_end_or_outcome_start():
    pd = pytest.importorskip("pandas")
    m = _builder()
    row = {field: None for field in m.DEPENDENCE_KEEP}
    row["candidate_id"] = "x"
    row["outcome_start_ts_ns"] = D0_START_NS
    row["overlap_window_end_ts_ns_15m"] = D0_START_NS + 900 * NS_PER_SECOND
    out = m.build_governance_linkage(pd.DataFrame([row]))
    assert "outcome_start_ts_ns" not in out.columns
    assert "overlap_window_end_ts_ns_15m" not in out.columns


def test_contract_freezes_outcome_firewall_and_family_deferrals():
    contract = json.loads((REPO_ROOT / "research" / "governance" / "market_state_contract_v1.json").read_text(encoding="utf-8"))
    assert contract["outcome_firewall"]["future_outcome_values_read"] is False
    assert contract["protected_data"]["p1_read"] is False
    assert contract["families"]["PERPS_TRADES"]["status"] == "IMPLEMENTED"
    assert contract["families"]["ORDER_BOOK"]["status"] == "DEFERRED"


def test_builder_source_declares_no_future_outcome_open():
    text = (REPO_ROOT / "research" / "r0" / "market_state" / "build_d0_market_state_store.py").read_text(encoding="utf-8")
    assert '"r0_3_future_outcomes_opened": False' in text
    assert "future/outcomes.parquet" not in text
    assert "candidate_decision_outcomes.parquet" not in text


def test_storage_grid_is_not_declared_a_scientific_feature_definition():
    contract = json.loads((REPO_ROOT / "research" / "governance" / "market_state_contract_v1.json").read_text(encoding="utf-8"))
    assert contract["trade_storage_grid"]["scientific_feature_definition"] is False


def test_pyarrow_trade_loader_rejects_float_nanosecond_schema(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")
    m = _builder()
    root = tmp_path / "canonical"
    day = root / "perps" / "trades" / "date=2026-05-18"
    day.mkdir(parents=True)
    table = pa.Table.from_pandas(pd.DataFrame({
        "ts_recv_ns": [float(D0_START_NS + 1)],
        "symbol": ["BTCUSDT"], "price": [100.0], "quantity": [1.0], "buyer_is_maker": [True],
    }), preserve_index=False)
    pq.write_table(table, day / "part.parquet")
    with pytest.raises(RuntimeError, match="not exact int64"):
        m.load_trade_days(root, ["2026-05-18"])


def test_prepare_trade_arrays_has_deterministic_source_row_tie_keys():
    pd = pytest.importorskip("pandas")
    m = _builder()
    frame = pd.DataFrame({
        "ts_recv_ns": [D0_START_NS + 1, D0_START_NS + 1],
        "price": [100.0, 101.0],
        "quantity": [1.0, 2.0],
        "buyer_is_maker": [True, False],
        "_source_path": ["a.parquet", "a.parquet"],
        "_source_row_ordinal": [10, 11],
    })
    arr = m.prepare_trade_arrays(frame)
    assert arr["tie_keys"] is not None
    assert arr["tie_keys"][0] != arr["tie_keys"][1]


def test_partition_status_requires_all_d0_days_in_interval():
    m = _builder()
    day2_start = D0_START_NS + 24 * 60 * 60 * NS_PER_SECOND
    assert m.interval_partition_status(day2_start - 1, day2_start + 1, {"2026-05-19"}) == "SOURCE_PARTITION_MISSING"


def test_field_registry_covers_all_structural_bucket_columns():
    m = _builder()
    reg = m.field_registry()
    fields = set(reg.loc[reg["table"] == "trade_buckets", "field"].astype(str))
    required = {
        "candidate_id", "cutoff_ts_ns", "bucket_offset_sec", "bucket_start_ts_ns",
        "bucket_end_ts_ns", "bucket_width_sec", "source_status", "trade_count",
        "quantity_sum", "notional_sum", "buyer_maker_true_quantity",
        "buyer_maker_false_quantity", "buyer_maker_unknown_quantity",
        "first_trade_ts_ns", "last_trade_ts_ns",
    }
    assert required <= fields


def test_clock_timestamps_are_not_auto_feature_eligible():
    m = _builder()
    reg = m.field_registry()
    core = reg[reg["table"] == "candidate_core"].set_index("field")
    assert core.loc["candidate_policy_availability_ts_ns", "role"] == "QUALITY_ONLY"
    assert core.loc["anchor_ts_ns", "role"] == "QUALITY_ONLY"
    assert core.loc["information_available_ts_ns", "role"] == "QUALITY_ONLY"



def test_merge_half_open_intervals_is_io_only_and_deterministic():
    m = _builder()
    got = m.merge_half_open_intervals([(10, 20), (20, 30), (5, 8), (7, 9), (40, 41)])
    assert got == [(5, 9), (10, 30), (40, 41)]


def test_pruned_loader_reads_only_intersecting_row_groups(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")
    m = _builder()
    root = tmp_path / "canonical"
    day = root / "perps" / "trades" / "date=2026-05-18"
    day.mkdir(parents=True)
    base = D0_START_NS
    # Four row groups, deliberately far apart in time.
    df = pd.DataFrame({
        "ts_recv_ns": [
            base + 1 * NS_PER_SECOND, base + 2 * NS_PER_SECOND,
            base + 100 * NS_PER_SECOND, base + 101 * NS_PER_SECOND,
            base + 200 * NS_PER_SECOND, base + 201 * NS_PER_SECOND,
            base + 300 * NS_PER_SECOND, base + 301 * NS_PER_SECOND,
        ],
        "symbol": ["BTCUSDT"] * 8,
        "price": [100.0 + i for i in range(8)],
        "quantity": [1.0] * 8,
        "buyer_is_maker": [True, False] * 4,
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), day / "part.parquet", row_group_size=2)
    got, _, present, stats = m.load_trade_intervals(
        root,
        ["2026-05-18"],
        [(base + 99 * NS_PER_SECOND, base + 102 * NS_PER_SECOND)],
        file_index_cache={},
    )
    assert present == {"2026-05-18"}
    assert stats["row_groups_total"] == 4
    assert stats["row_groups_read"] == 1
    assert got["ts_recv_ns"].tolist() == [base + 100 * NS_PER_SECOND, base + 101 * NS_PER_SECOND]


def test_pruned_builder_falls_back_for_exact_latest_prior_semantics(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")
    m = _builder()
    root = tmp_path / "canonical"
    day = root / "perps" / "trades" / "date=2026-05-18"
    day.mkdir(parents=True)
    cutoff = D0_START_NS + 2 * 60 * 60 * NS_PER_SECOND
    old_trade = cutoff - 45 * 60 * NS_PER_SECOND  # older than coarse 30m storage history
    df = pd.DataFrame({
        "ts_recv_ns": [old_trade],
        "symbol": ["BTCUSDT"],
        "price": [123.0],
        "quantity": [2.0],
        "buyer_is_maker": [False],
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), day / "part.parquet")
    core = pd.DataFrame([{
        "candidate_id": "x",
        "candidate_date": "2026-05-18",
        "direction": 1,
        "candidate_policy_availability_ts_ns": cutoff - NS_PER_SECOND,
        "anchor_ts_ns": cutoff - 2 * NS_PER_SECOND,
        "information_available_ts_ns": cutoff,
        "decision_ts_ns": cutoff,
        "clock_available": True,
        "clock_unavailable_reason": None,
    }])
    state, fine, coarse, coverage, _, scan = m.build_trade_state_and_buckets(core, root, 1)
    assert len(state) == 1
    assert int(state.loc[0, "latest_trade_ts_ns"]) == old_trade
    assert float(state.loc[0, "latest_trade_price"]) == 123.0
    assert scan["full_day_fallback_groups"] == 1
    assert len(fine) == 60 and len(coarse) == 30
    assert int(fine["trade_count"].sum()) == 0
    assert int(coarse["trade_count"].sum()) == 0
