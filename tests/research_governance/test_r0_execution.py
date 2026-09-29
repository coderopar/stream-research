from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT=Path(__file__).resolve().parents[2]
SRC=REPO_ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))

from research_core.clocks.contract import D0_START_NS, NS_PER_SECOND
from research_core.execution import (
    CostAssumptions, LatencyAssumptions, ReplayBook, ZERO_LATENCY_AUDIT,
    build_execution_timeline, simulate_marketable_fill, total_cost_bps,
)


def _builder():
    p=REPO_ROOT/"research"/"r0"/"execution"/"build_d0_execution_foundation.py"
    spec=importlib.util.spec_from_file_location("r06_builder",p); assert spec and spec.loader
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m


def test_latency_components_are_explicit_and_monotone():
    a=LatencyAssumptions(1,2,3,4,5,6,"x"); t=build_execution_timeline(D0_START_NS,a)
    vals=[t[x] for x in ("information_available_ts_ns","computation_complete_ts_ns","decision_ts_ns","order_send_ts_ns","estimated_exchange_arrival_ts_ns","executable_entry_ts_ns","fill_complete_ts_ns")]
    assert vals==sorted(vals) and vals[-1]==D0_START_NS+21


def test_negative_latency_is_rejected():
    with pytest.raises(ValueError): LatencyAssumptions(0,0,0,-1,0,0)


def test_zero_latency_profile_is_explicitly_audit_only():
    assert ZERO_LATENCY_AUDIT.profile_id=="ZERO_LATENCY_AUDIT_ONLY"


def _bootstrapped_book():
    b=ReplayBook(price_scale=10,accelerated=False); t=D0_START_NS+10*NS_PER_SECOND
    b.apply_snapshot({"ts_recv_ns":t,"last_update_id":10,"bids":[{"price":"99.9","quantity":"20"},{"price":"99.8","quantity":"20"}],"asks":[{"price":"100.1","quantity":"20"},{"price":"100.2","quantity":"20"}]})
    b.apply_depth({"ts_recv_ns":t+1,"first_update_id":11,"final_update_id":11,"previous_final_update_id":10,"bid_updates":[],"ask_updates":[]})
    return b,t


def test_snapshot_is_not_executable_until_exact_bridge():
    b=ReplayBook(price_scale=10,accelerated=False); t=D0_START_NS+10
    b.apply_snapshot({"ts_recv_ns":t,"last_update_id":7,"bids":[{"price":"99.9","quantity":"1"}],"asks":[{"price":"100.1","quantity":"1"}]})
    assert b.state(t).status=="BRIDGE_PENDING" and not b.state(t).sequence_valid


def test_exact_pu_bridge_bootstraps_book():
    b,t=_bootstrapped_book(); s=b.state(t+1)
    assert s.sequence_valid and s.best_bid==99.9 and s.best_ask==100.1 and s.last_update_id==11


def test_bad_bootstrap_bridge_fails_closed():
    b=ReplayBook(price_scale=10,accelerated=False); t=D0_START_NS+10
    b.apply_snapshot({"ts_recv_ns":t,"last_update_id":7,"bids":[{"price":"99.9","quantity":"1"}],"asks":[{"price":"100.1","quantity":"1"}]})
    b.apply_depth({"ts_recv_ns":t+1,"first_update_id":8,"final_update_id":8,"previous_final_update_id":6,"bid_updates":[],"ask_updates":[]})
    assert not b.state(t+1).sequence_valid and b.state(t+1).status=="BOOTSTRAP_LOCAL_BRIDGE_FAILED"


def test_sequence_gap_invalidates_healthy_book():
    b,t=_bootstrapped_book(); b.apply_depth({"ts_recv_ns":t+2,"first_update_id":12,"final_update_id":12,"previous_final_update_id":999,"bid_updates":[],"ask_updates":[]})
    assert not b.state(t+2).sequence_valid and b.state(t+2).status=="PU_SEQUENCE_GAP"


def test_zero_quantity_deletes_level():
    b,t=_bootstrapped_book(); b.apply_depth({"ts_recv_ns":t+2,"first_update_id":12,"final_update_id":12,"previous_final_update_id":11,"bid_updates":[{"price":"99.9","quantity":"0"}],"ask_updates":[]})
    assert b.state(t+2).best_bid==99.8


def test_crossed_reconstructed_book_invalidates():
    b,t=_bootstrapped_book(); b.apply_depth({"ts_recv_ns":t+2,"first_update_id":12,"final_update_id":12,"previous_final_update_id":11,"bid_updates":[{"price":"100.2","quantity":"1"}],"ask_updates":[]})
    assert not b.state(t+2).sequence_valid and b.state(t+2).status=="CROSSED_BOOK"


def test_marketable_buy_walks_asks_and_sell_walks_bids():
    b,t=_bootstrapped_book(); s=b.state(t+1)
    buy=simulate_marketable_fill(side="BUY",quote_notional=2500,levels=b.levels_for("BUY"),arrival_mid=s.mid)
    sell=simulate_marketable_fill(side="SELL",quote_notional=2500,levels=b.levels_for("SELL"),arrival_mid=s.mid)
    assert buy.complete and sell.complete and buy.vwap>s.mid and sell.vwap<s.mid
    assert buy.raw_crossing_cost_bps>0 and sell.raw_crossing_cost_bps>0


def test_insufficient_visible_depth_never_extrapolates():
    b,t=_bootstrapped_book(); s=b.state(t+1)
    f=simulate_marketable_fill(side="BUY",quote_notional=1_000_000,levels=b.levels_for("BUY"),arrival_mid=s.mid)
    assert not f.complete and f.filled_quote<1_000_000 and f.raw_crossing_cost_bps is None


def test_total_cost_requires_explicit_cost_object_and_adds_components():
    assert total_cost_bps(2.0,CostAssumptions(4.0,1.0,-0.5,"x"))==pytest.approx(6.5)


def test_negative_taker_fee_rejected():
    with pytest.raises(ValueError): CostAssumptions(-1,0,0)


def test_builder_refuses_p1_family_path():
    m=_builder()
    from datetime import date
    with pytest.raises(RuntimeError,match="non-D0"): m.canonical_family_paths(Path("x"),"depth_updates",date(2026,7,19))


def test_builder_audit_row_preserves_clock_unavailable_candidate():
    m=_builder(); b=ReplayBook(price_scale=10,accelerated=False)
    r=m._audit_row({"candidate_id":"x","candidate_date":"2026-05-18","direction":1,"clock_available":False,"information_available_ts_ns":None},b)
    assert r["book_status"]=="CLOCK_UNAVAILABLE" and r["arrival_ts_ns"] is None


def test_contract_forbids_passive_fill_assumption_and_outcome_tuning():
    c=json.loads((REPO_ROOT/"research"/"governance"/"execution_contract_v1.json").read_text())
    assert c["marketable_execution"]["passive_fill_assumptions"] is False
    assert c["outcome_firewall"]["future_outcome_values_read"] is False
    assert c["outcome_firewall"]["latency_or_cost_tuning_from_outcomes"] is False


def test_contract_marks_audit_notional_and_latency_non_strategy():
    c=json.loads((REPO_ROOT/"research"/"governance"/"execution_contract_v1.json").read_text())
    assert c["audit_only"]["scientific_strategy_definition"] is False
    assert c["audit_only"]["arrival_profile"]=="ZERO_LATENCY_AUDIT_ONLY"


def test_builder_source_does_not_open_future_outcomes():
    text=(REPO_ROOT/"research"/"r0"/"execution"/"build_d0_execution_foundation.py").read_text()
    assert '"future_outcomes_opened":False' in text
    assert "future/outcomes.parquet" not in text
    assert "candidate_decision_outcomes.parquet" not in text


def test_fill_at_arrival_uses_opposite_side_visible_depth():
    f=simulate_marketable_fill(side="BUY",quote_notional=100,levels=[(101,1)],arrival_mid=100)
    assert f.vwap==101 and f.filled_quote==100 and f.complete


def test_state_age_is_recorded_not_thresholded():
    b,t=_bootstrapped_book(); s=b.state(t+10_000_000)
    assert s.sequence_valid and s.state_age_ns==9_999_999


def test_snapshot_during_healthy_replay_does_not_reset_book():
    b,t=_bootstrapped_book(); before=b.state(t+1)
    b.apply_snapshot({"ts_recv_ns":t+2,"last_update_id":999,"bids":[{"price":"90","quantity":"1"}],"asks":[{"price":"91","quantity":"1"}]})
    after=b.state(t+2)
    assert after.sequence_valid and after.best_bid==before.best_bid and after.best_ask==before.best_ask


def test_fill_reference_cost_is_mid_based_and_side_symmetric_at_touch():
    buy=simulate_marketable_fill(side="BUY",quote_notional=100,levels=[(100.1,2)],arrival_mid=100)
    sell=simulate_marketable_fill(side="SELL",quote_notional=100,levels=[(99.9,2)],arrival_mid=100)
    assert buy.raw_crossing_cost_bps==pytest.approx(10.0)
    assert sell.raw_crossing_cost_bps==pytest.approx(10.0)


def test_accelerated_replay_matches_python_reference_when_available():
    from research_core.execution import acceleration_available
    if not acceleration_available():
        pytest.skip("numba unavailable")
    t=D0_START_NS+50
    snap={"ts_recv_ns":t,"last_update_id":10,"bids":[{"price":"99.9","quantity":"2"}],"asks":[{"price":"100.1","quantity":"3"}]}
    d1={"ts_recv_ns":t+1,"first_update_id":11,"final_update_id":11,"previous_final_update_id":10,"bid_updates":[],"ask_updates":[]}
    d2={"ts_recv_ns":t+2,"first_update_id":12,"final_update_id":12,"previous_final_update_id":11,"bid_updates":[{"price":"99.8","quantity":"4"}],"ask_updates":[{"price":"100.2","quantity":"5"}]}
    py=ReplayBook(price_scale=10,accelerated=False); fast=ReplayBook(price_scale=10,accelerated=True)
    for b in (py,fast):
        b.apply_snapshot(snap); b.apply_depth(d1); b.apply_depth(d2)
    a=py.state(t+2); z=fast.state(t+2)
    assert (a.sequence_valid,a.best_bid,a.best_ask,a.mid,a.last_update_id)==(z.sequence_valid,z.best_bid,z.best_ask,z.mid,z.last_update_id)


def test_pyarrow_depth_loader_rejects_float_nanosecond_schema(tmp_path: Path):
    pa=pytest.importorskip("pyarrow"); pq=pytest.importorskip("pyarrow.parquet")
    m=_builder(); p=tmp_path/"depth.parquet"
    level=pa.list_(pa.struct([pa.field("price",pa.string()),pa.field("quantity",pa.string())]))
    schema=pa.schema([
        pa.field("ts_recv_ns",pa.float64()),pa.field("symbol",pa.string()),
        pa.field("first_update_id",pa.int64()),pa.field("final_update_id",pa.int64()),pa.field("previous_final_update_id",pa.int64()),
        pa.field("bid_updates",level),pa.field("ask_updates",level),pa.field("source_relative_path",pa.string()),pa.field("source_record_number",pa.int64())])
    table=pa.Table.from_pylist([{"ts_recv_ns":float(D0_START_NS),"symbol":"BTCUSDT","first_update_id":1,"final_update_id":1,"previous_final_update_id":0,"bid_updates":[],"ask_updates":[],"source_relative_path":"x","source_record_number":1}],schema=schema)
    pq.write_table(table,p)
    with pytest.raises(RuntimeError,match="must be exact int64"):
        list(m._iter_depth_file(p,10.0))


def test_equal_receive_time_orders_snapshot_before_depth(tmp_path: Path):
    pa=pytest.importorskip("pyarrow"); pq=pytest.importorskip("pyarrow.parquet")
    m=_builder(); level=pa.list_(pa.struct([pa.field("price",pa.string()),pa.field("quantity",pa.string())])); ts=D0_START_NS+123
    sp=tmp_path/"snap.parquet"; dp=tmp_path/"depth.parquet"
    ss=pa.schema([pa.field("ts_recv_ns",pa.int64()),pa.field("symbol",pa.string()),pa.field("last_update_id",pa.int64()),pa.field("bids",level),pa.field("asks",level),pa.field("source_relative_path",pa.string()),pa.field("source_record_number",pa.int64())])
    ds=pa.schema([pa.field("ts_recv_ns",pa.int64()),pa.field("symbol",pa.string()),pa.field("first_update_id",pa.int64()),pa.field("final_update_id",pa.int64()),pa.field("previous_final_update_id",pa.int64()),pa.field("bid_updates",level),pa.field("ask_updates",level),pa.field("source_relative_path",pa.string()),pa.field("source_record_number",pa.int64())])
    pq.write_table(pa.Table.from_pylist([{"ts_recv_ns":ts,"symbol":"BTCUSDT","last_update_id":10,"bids":[{"price":"99.9","quantity":"1"}],"asks":[{"price":"100.1","quantity":"1"}],"source_relative_path":"s","source_record_number":1}],schema=ss),sp)
    pq.write_table(pa.Table.from_pylist([{"ts_recv_ns":ts,"symbol":"BTCUSDT","first_update_id":11,"final_update_id":11,"previous_final_update_id":10,"bid_updates":[],"ask_updates":[],"source_relative_path":"d","source_record_number":1}],schema=ds),dp)
    ev=list(m.merged_book_events([sp],[dp],10.0))
    assert [x["_kind"] for x in ev]==["SNAPSHOT","DEPTH"]


def test_r03_label_clock_is_not_frozen_as_future_strategy_entry():
    c=json.loads((REPO_ROOT/"research"/"governance"/"execution_contract_v1.json").read_text())
    assert c["future_strategy_outcomes"]["r0_3_label_clock_is_entry_substitute"] is False
    assert "executable_entry_ts_ns" in c["future_strategy_outcomes"]["rule"]
