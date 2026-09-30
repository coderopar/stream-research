from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_core.dry_run import (  # noqa: E402
    DRY_RUN_FEATURE_WINDOW_SEC,
    DRY_RUN_HORIZON_SEC,
    add_mechanical_net_proxy,
    apply_flat_only_position_state_machine,
    attach_future_observability,
    build_causal_analysis_population,
    build_fixed_test_state,
    dependence_aware_contrast,
)
from research_core.registry import (  # noqa: E402
    RegistryValidationError,
    validate_confirmation_batch,
    validate_evidence_budget,
    validate_event_history,
    validate_protected_cohort,
    validate_registered_spec,
)


def _runner():
    p = REPO_ROOT / "research" / "r0" / "dry_run" / "run_full_d0_dry_run.py"
    spec = importlib.util.spec_from_file_location("r08_runner", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _core():
    return pd.DataFrame([
        {"candidate_id": "A", "direction": 1, "information_available_ts_ns": 10_000_000_000, "clock_available": True},
        {"candidate_id": "B", "direction": -1, "information_available_ts_ns": 100_000_000_000, "clock_available": True},
        {"candidate_id": "C", "direction": 1, "information_available_ts_ns": 400_000_000_000, "clock_available": True},
        {"candidate_id": "D", "direction": -1, "information_available_ts_ns": 800_000_000_000, "clock_available": False},
    ])


def _buckets(missing_for: str | None = None):
    rows = []
    for cid, state in (("A", True), ("B", True), ("C", False), ("D", True)):
        for off in range(-60, 0):
            rows.append({
                "candidate_id": cid,
                "bucket_offset_sec": off,
                "source_status": "SOURCE_PARTITION_MISSING" if cid == missing_for and off == -1 else "AVAILABLE",
                "buyer_maker_true_quantity": 1.0 if state else 2.0,
                "buyer_maker_false_quantity": 2.0 if state else 1.0,
            })
    return pd.DataFrame(rows)


def _coverage():
    return pd.DataFrame([{"candidate_id": x, "fine_history_status": "AVAILABLE"} for x in "ABCD"])


def _execution():
    return pd.DataFrame([
        {"candidate_id": x, "book_status": "AVAILABLE", "buy_fill_complete": True, "sell_fill_complete": True,
         "buy_raw_crossing_cost_bps": 1.0, "sell_raw_crossing_cost_bps": 1.5}
        for x in "ABCD"
    ])


def _outcomes(b_status="AVAILABLE"):
    return pd.DataFrame([
        {"candidate_id": "A", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": 8.0, "mae_from_decision_bps": 3.0},
        {"candidate_id": "B", "outcome_horizon_sec": 300, "outcome_status": b_status, "outcome_observable": b_status == "AVAILABLE", "directional_return_to_last_trade_bps": 2.0 if b_status == "AVAILABLE" else None, "mae_from_decision_bps": 4.0 if b_status == "AVAILABLE" else None},
        {"candidate_id": "C", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": -1.0, "mae_from_decision_bps": 6.0},
        {"candidate_id": "D", "outcome_horizon_sec": 300, "outcome_status": "CLOCK_UNAVAILABLE", "outcome_observable": False, "directional_return_to_last_trade_bps": None, "mae_from_decision_bps": None},
    ])


def _dep():
    return pd.DataFrame([
        {"candidate_id": "A", "conservative_inference_cluster_id": "K1"},
        {"candidate_id": "B", "conservative_inference_cluster_id": "K1"},
        {"candidate_id": "C", "conservative_inference_cluster_id": "K1"},
        {"candidate_id": "D", "conservative_inference_cluster_id": "K2"},
    ])


def _attached(b_status="AVAILABLE"):
    state = build_fixed_test_state(_buckets())
    pop = build_causal_analysis_population(_core(), _coverage(), _execution(), state)
    return attach_future_observability(pop, _outcomes(b_status), _dep())


def test_fixed_test_state_uses_exact_trailing_ten_buckets():
    state = build_fixed_test_state(_buckets())
    assert DRY_RUN_FEATURE_WINDOW_SEC == 10
    a = state.set_index("candidate_id").loc["A"]
    c = state.set_index("candidate_id").loc["C"]
    assert bool(a["feature_available"]) and bool(a["dry_run_state_true"])
    assert bool(c["feature_available"]) and not bool(c["dry_run_state_true"])
    assert a["raw_buyer_maker_false_qty_10s"] == 20.0
    assert a["raw_buyer_maker_true_qty_10s"] == 10.0


def test_feature_missingness_fails_closed_not_zero_imputed():
    state = build_fixed_test_state(_buckets(missing_for="A")).set_index("candidate_id")
    assert not bool(state.loc["A", "feature_available"])
    assert not bool(state.loc["A", "dry_run_state_true"])


def test_causal_eligibility_does_not_require_future_outcome_fields():
    state = build_fixed_test_state(_buckets())
    pop = build_causal_analysis_population(_core(), _coverage(), _execution(), state)
    assert "outcome_status" not in pop.columns
    assert bool(pop.set_index("candidate_id").loc["A", "trading_eligible"])
    assert not bool(pop.set_index("candidate_id").loc["D", "trading_eligible"])


def test_future_observability_cannot_change_eligibility():
    state = build_fixed_test_state(_buckets())
    pop = build_causal_analysis_population(_core(), _coverage(), _execution(), state)
    before = pop.set_index("candidate_id")["trading_eligible"].to_dict()
    attached = attach_future_observability(pop, _outcomes("D0_BOUNDARY_CENSOR"), _dep())
    after = attached.set_index("candidate_id")["trading_eligible"].to_dict()
    assert before == after
    assert bool(after["B"]) and not bool(attached.set_index("candidate_id").loc["B", "analysis_observed"])


def test_wrong_horizon_is_rejected():
    bad = _outcomes().copy(); bad["outcome_horizon_sec"] = 600
    state = build_fixed_test_state(_buckets())
    pop = build_causal_analysis_population(_core(), _coverage(), _execution(), state)
    with pytest.raises(ValueError, match="5-minute"):
        attach_future_observability(pop, bad, _dep())


def test_position_machine_prevents_overlapping_exposure():
    positioned = apply_flat_only_position_state_machine(_attached())
    by_id = positioned.set_index("candidate_id")
    assert bool(by_id.loc["A", "position_entered"])
    assert by_id.loc["B", "position_action"] == "IGNORED_WHILE_POSITION_OPEN"
    assert by_id.loc["C", "position_action"] == "NO_SIGNAL"
    assert not bool(by_id.loc["D", "position_entered"])


def test_position_horizon_is_fixed_five_minutes():
    positioned = apply_flat_only_position_state_machine(_attached())
    a = positioned.set_index("candidate_id").loc["A"]
    assert int(a["planned_exit_ts_ns"]) - int(a["information_available_ts_ns"]) == DRY_RUN_HORIZON_SEC * 1_000_000_000


def test_mechanical_proxy_subtracts_direction_specific_entry_crossing_cost():
    out = add_mechanical_net_proxy(apply_flat_only_position_state_machine(_attached())).set_index("candidate_id")
    assert out.loc["A", "dry_run_mechanical_net_proxy_bps"] == pytest.approx(7.0)
    assert out.loc["B", "dry_run_mechanical_net_proxy_bps"] == pytest.approx(0.5)


def test_dependence_contrast_requires_within_cluster_support():
    f = add_mechanical_net_proxy(_attached())
    c = dependence_aware_contrast(f)
    assert c.independent_units == 1
    assert c.status == "INDETERMINATE_SINGLE_UNIT"


def test_dry_run_registry_packet_validates_but_is_non_promotable():
    m = _runner()
    rec = m.make_dry_run_registry_records({"parent": "0" * 64}, "abc1234")
    validate_evidence_budget(rec["budget"])
    validate_registered_spec(rec["hypothesis"])
    validate_protected_cohort(rec["cohort"])
    validate_confirmation_batch(rec["batch"])
    validate_event_history(rec["events"])
    assert rec["events"][-1]["classification"] == "SCIENTIFIC_ONLY"
    assert "No D1 may ever be spent" in rec["budget"]["do_not_spend_threshold"]


def test_forbidden_action_matrix_all_fails_closed():
    m = _runner(); rec = m.make_dry_run_registry_records({"parent": "0" * 64}, "abc1234")
    rows = m.forbidden_action_matrix(rec)
    assert rows and all(r["blocked"] for r in rows)
    names = {r["check"] for r in rows}
    assert "P1 outcome access before real D1 freeze" in names
    assert "future-derived eligibility" in names
    assert "random cross-validation" in names
    assert "same-version material mutation" in names
    assert "terminal-version rescue" in names


def test_isolated_registry_rehearsal_is_deterministic(tmp_path: Path):
    m = _runner(); rec = m.make_dry_run_registry_records({"parent": "0" * 64}, "abc1234")
    a = m.write_isolated_registries(tmp_path / "a", rec)
    b = m.write_isolated_registries(tmp_path / "b", rec)
    assert a == b


def test_production_registry_snapshot_detects_any_mutation(tmp_path: Path):
    m = _runner()
    for name in m.PRODUCTION_REGISTRY_FILES:
        (tmp_path / name).write_text("", encoding="utf-8")
    before = m.snapshot_production_registries(tmp_path)
    (tmp_path / m.PRODUCTION_REGISTRY_FILES[0]).write_text("x", encoding="utf-8")
    after = m.snapshot_production_registries(tmp_path)
    assert before != after


def test_p1_seal_check_requires_outcomes_unseen(tmp_path: Path):
    m = _runner(); p = tmp_path / "protected.yaml"
    p.write_text("pool_id: P1_PROTECTED\nstatus: SEALED\noutcome_exposure_status: UNSEEN\n", encoding="utf-8")
    m.assert_p1_sealed(p)
    p.write_text("pool_id: P1_PROTECTED\nstatus: SEALED\noutcome_exposure_status: SEEN\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="P1 seal"):
        m.assert_p1_sealed(p)


def test_dataframe_substantive_hash_ignores_row_order_not_values():
    m = _runner(); f = pd.DataFrame([{"candidate_id": "B", "x": 2.0}, {"candidate_id": "A", "x": 1.0}])
    g = f.iloc[::-1].reset_index(drop=True)
    assert m.dataframe_substantive_hash(f, ["candidate_id", "x"]) == m.dataframe_substantive_hash(g, ["candidate_id", "x"])
    g.loc[g["candidate_id"] == "A", "x"] = 9.0
    assert m.dataframe_substantive_hash(f, ["candidate_id", "x"]) != m.dataframe_substantive_hash(g, ["candidate_id", "x"])


def test_runner_source_does_not_reference_p1_market_paths():
    text = (REPO_ROOT / "research/r0/dry_run/run_full_d0_dry_run.py").read_text(encoding="utf-8")
    assert "data/canonical" not in text
    assert "P1_PROTECTED" in text  # guardrail negative control only
    assert "p1_read\": False" in text


def test_contract_declares_non_promotable_and_production_registry_isolation():
    contract = json.loads((REPO_ROOT / "research/governance/r0_dry_run_contract_v1.json").read_text(encoding="utf-8"))
    assert contract["non_promotable"] is True
    assert contract["production_registries_must_remain_unchanged"] is True
    assert contract["protected_data"]["p1_market_data_read"] is False
    assert contract["eligibility"]["future_fields_allowed"] is False


def test_r08_readme_requires_research_readme_update():
    text = (REPO_ROOT / "research/r0/dry_run/README.md").read_text(encoding="utf-8")
    assert "research/README.md" in text
    assert "SCIENTIFIC_ONLY" in text
    assert "outcome observability" in text.lower()


def test_self_test_passes_without_market_files():
    assert _runner().self_test() == 0



def test_worst_mae_uses_maximum_nonnegative_magnitude():
    source = (REPO_ROOT / "research/r0/dry_run/run_full_d0_dry_run.py").read_text(encoding="utf-8")
    assert 'observed_mae.max()' in source
    assert 'observed_mae < 0' in source


def test_full_runner_tiny_pyarrow_fixture(tmp_path: Path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    m = _runner()

    def write_parquet(path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        return m.sha256_file(path)

    r03 = tmp_path / "r03"; r04 = tmp_path / "r04"; r05 = tmp_path / "r05"; r06 = tmp_path / "r06"
    outcomes = [
        {"candidate_id": "A", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": 8.0, "mae_from_decision_bps": 3.0},
        {"candidate_id": "B", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": 2.0, "mae_from_decision_bps": 4.0},
        {"candidate_id": "C", "outcome_horizon_sec": 300, "outcome_status": "AVAILABLE", "outcome_observable": True, "directional_return_to_last_trade_bps": -1.0, "mae_from_decision_bps": 6.0},
        {"candidate_id": "D", "outcome_horizon_sec": 300, "outcome_status": "CLOCK_UNAVAILABLE", "outcome_observable": False, "directional_return_to_last_trade_bps": None, "mae_from_decision_bps": None},
    ]
    dep = [{"candidate_id": x, "conservative_inference_cluster_id": "K1" if x != "D" else "K2"} for x in "ABCD"]
    core = _core().to_dict("records")
    buckets = _buckets().to_dict("records")
    cov = _coverage().to_dict("records")
    exe = _execution().to_dict("records")

    out_p = r03 / "future/outcomes.parquet"; dep_p = r04 / "dependence/candidate_dependence.parquet"
    core_p = r05 / "causal/candidate_core.parquet"; bucket_p = r05 / "causal/trade_buckets_1s.parquet"; cov_p = r05 / "quality/trade_coverage.parquet"
    exe_p = r06 / "execution/audit_zero_latency_state.parquet"
    hashes = {
        "out": write_parquet(out_p, outcomes), "dep": write_parquet(dep_p, dep),
        "core": write_parquet(core_p, core), "buckets": write_parquet(bucket_p, buckets),
        "cov": write_parquet(cov_p, cov), "exe": write_parquet(exe_p, exe),
    }
    manifests = [
        (r03, {"schema_name": "r0_corrected_d0_substrate_v1", "counts": {"candidate_rows": 4}, "outputs": {"outcomes": {"sha256": hashes["out"]}}}),
        (r04, {"schema_name": "r0_dependence_framework_v1", "outputs": {"candidate_dependence": {"sha256": hashes["dep"]}}}),
        (r05, {"schema_name": "r0_causal_market_state_v1", "counts": {"governed_clock_rows": 3}, "outputs": {"candidate_core": {"sha256": hashes["core"]}, "trade_buckets_1s": {"sha256": hashes["buckets"]}, "coverage": {"sha256": hashes["cov"]}}}),
        (r06, {"schema_name": "r0_execution_simulator_v1", "outputs": {"audit_execution_state": {"sha256": hashes["exe"]}}}),
    ]
    for root, manifest in manifests:
        p = root / "metadata/manifest.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(manifest), encoding="utf-8")

    regs = tmp_path / "prod_regs"; regs.mkdir()
    for name in m.PRODUCTION_REGISTRY_FILES: (regs / name).write_text("", encoding="utf-8")
    protected = tmp_path / "protected.yaml"
    protected.write_text("pool_id: P1_PROTECTED\nstatus: SEALED\noutcome_exposure_status: UNSEEN\n", encoding="utf-8")
    outroot = tmp_path / "out"; reports = tmp_path / "reports"
    monkeypatch.setattr(m, "EXPECTED_CANDIDATES", 4); monkeypatch.setattr(m, "EXPECTED_GOVERNED", 3)
    args = type("Args", (), {"r03_root": r03, "r04_root": r04, "r05_root": r05, "r06_root": r06, "production_registry_root": regs, "protected_register": protected, "output_root": outroot, "report_root": reports})()
    assert m.run(args) == 0
    manifest = json.loads((outroot / "metadata/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "BUILD_COMPLETE_REVIEW_REQUIRED"
    assert manifest["summary"]["candidate_rows"] == 4
    # R0.3 stores MAE as a non-negative adverse-excursion magnitude, so
    # worst entered-position risk is the maximum among positions actually
    # entered. This fixture enters only A (MAE 3.0); B (MAE 4.0) is ignored
    # while A is open and therefore must not contaminate position risk.
    assert manifest["summary"]["positions_entered"] == 1
    assert manifest["summary"]["ignored_while_position_open"] == 1
    assert manifest["summary"]["position_worst_mae_bps"] == 3.0
    assert manifest["summary"]["production_registry_mutations"] == 0
    assert manifest["summary"]["p1_read"] is False
    first_hash = manifest["substantive_run_hash"]
    assert m.run(args) == 0
    again = json.loads((outroot / "metadata/manifest.json").read_text(encoding="utf-8"))
    assert again["substantive_run_hash"] == first_hash


def test_idempotence_rebaselines_after_implementation_change_but_not_same_implementation():
    m = _runner()
    parents = {"r03": {"sha256": "p"}}
    # A pre-fingerprint or differently fingerprinted historical manifest is a prior
    # implementation baseline and must not block a legitimate code/spec correction.
    m.enforce_same_implementation_idempotence(
        {"parents": parents, "substantive_run_hash": "old"},
        current_implementation_sha256="new-impl",
        parent_hashes=parents,
        substantive_run_hash="new",
    )
    m.enforce_same_implementation_idempotence(
        {"parents": parents, "implementation_fingerprint": {"sha256": "old-impl"}, "substantive_run_hash": "old"},
        current_implementation_sha256="new-impl",
        parent_hashes=parents,
        substantive_run_hash="new",
    )
    # But the exact same implementation and parents changing substantive output is
    # genuine nondeterminism and must fail closed.
    with pytest.raises(RuntimeError, match="same-input same-implementation"):
        m.enforce_same_implementation_idempotence(
            {"parents": parents, "implementation_fingerprint": {"sha256": "same"}, "substantive_run_hash": "old"},
            current_implementation_sha256="same",
            parent_hashes=parents,
            substantive_run_hash="new",
        )
