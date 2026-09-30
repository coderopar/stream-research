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

from research_core.registry import (  # noqa: E402
    RegistryValidationError,
    ReproducibilityError,
    append_jsonl,
    assert_append_only_prefix,
    assert_blind_design_eligible,
    assert_eligibility_fields_safe,
    assert_predictor_columns_safe,
    assert_substantive_equal,
    assert_time_respecting_validation,
    authorize_protected_access,
    merge_contamination,
    read_jsonl,
    validate_access_event_history,
    validate_confirmation_batch,
    validate_decision_log,
    validate_evidence_budget,
    validate_event_history,
    validate_genealogy,
    validate_hypothesis_version,
    validate_material_change,
    validate_protected_cohort,
    validate_registered_spec,
    verify_freeze_hash,
    with_freeze_hash,
)


def _validator():
    p = REPO_ROOT / "research" / "r0" / "registries" / "validate_research_governance.py"
    spec = importlib.util.spec_from_file_location("r07_validator", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _hyp(version: int = 1, **overrides):
    row = {
        "hypothesis_id": "HYP-TEST001",
        "version": version,
        "conceptual_family": "FAM-TEST001",
        "creation_source": "D0_DISCOVERY",
        "created_at_utc": "2026-09-29T00:00:00Z",
        "scientific_proposition": "A causally observable state changes executable future return distribution.",
        "trading_proposition": "If translated successfully, enter the registered position.",
        "relationship_type": "MARGINAL",
        "parent_hypothesis_ids": [],
        "derived_from_results": ["D0_ONLY"],
        "datasets_that_influenced_design": ["D0"],
        "material_spec": {"market_states": ["x"]},
    }
    row.update(overrides)
    return with_freeze_hash(row)


def _registered_hyp():
    spec = {
        "market_states": ["state_x"],
        "information_clocks": {"state_x": "information_available_ts_ns"},
        "eligibility_rules": {"clock_available": True},
        "decision_clock": "max information_available_ts_ns",
        "latency_model": "LATENCY-PROFILE-001",
        "entry_rule": "marketable depth walk at arrival",
        "position_state_machine": "PSM-001",
        "research_notional": "fixed standardized notional",
        "exit_rule": "frozen time exit",
        "risk_rules": ["max one concurrent position"],
        "cost_model": "COST-001",
        "primary_outcome": "net executable return",
        "primary_economic_metric": "mean net return",
        "registered_benchmarks": ["PARENT-STRATEGY-001"],
        "minimum_economically_meaningful_effect": "pre-specified positive incremental edge",
        "risk_gates": ["tail-loss gate"],
        "inference_plan": "cluster by frozen dependence unit",
        "dependence_episode_definition": {
            "deterministic": True,
            "primary_inference_unit": "conservative frozen cluster",
            "plausible_alternatives": ["day", "overlap component"],
            "governing_conservative_rule": "coarsest registered plausible result governs",
        },
        "evidence_budget_id": "EB-TEST001",
        "minimum_support": "at least registered independent episodes",
        "independent_support_precision_plan": {
            "effect_assumption": "conservatively shrunk D0 effect",
            "do_not_spend_threshold": "minimum 30 independent episodes",
        },
        "primary_failure_criterion": "registered return/MEE or risk gate fails",
        "d1_feasibility_plan": {
            "permitted_gate_outputs": ["PASS", "FAIL"],
            "detailed_access_restrictions": "quarantined from design",
            "bounded_fallback_tree": [],
            "same_cohort_redesign_forbidden": True,
        },
        "post_unblinding_defect_rule": "material defect invalidates affected confirmation",
        "code_commit": "abc1234",
        "artifact_hashes": {"config": "0" * 64},
        "confirmation_plan": {
            "confirmation_batch_id": "CB-TEST001",
            "protected_cohort_id": "D1-TEST001",
            "protocol_version": "1.2",
            "immutable_d1_data_version": "d1-v1",
            "unblinding_rule": "one-use frozen confirmation run",
        },
    }
    return _hyp(
        material_spec=spec,
        adversarial_review={
            "review_id": "AR-TEST001",
            "reviewer_context_id": "CTX-FRESH001",
            "frozen_packet_hash": "1" * 64,
            "fresh_context": True,
            "outcome_blind": True,
            "participated_in_discovery": False,
            "unresolved_objections": [],
            "objection_dispositions": [],
        },
    )


def _budget(**overrides):
    row = {
        "evidence_budget_id": "EB-TEST001",
        "family_ids": ["FAM-TEST001"],
        "primary_claim_ids": ["HYP-TEST001@1"],
        "secondary_claim_ids": [],
        "robustness_diagnostic_ids": ["ROBUST-001"],
        "control_method": "SINGLE_PRIMARY_DECISION",
        "operating_guarantee": "One frozen primary decision; no alternative unallocated success path.",
        "anti_family_splitting_rationale": "All threshold/horizon variants of this mechanism remain in one family.",
        "expected_independent_support": {"unit": "episodes", "expected": 40},
        "do_not_spend_threshold": "do not spend D1 if expected independent support is below 30",
        "unallocated_alternative_success_paths": False,
    }
    row.update(overrides)
    return with_freeze_hash(row)


def _batch(**overrides):
    row = {
        "confirmation_batch_id": "CB-TEST001",
        "protocol_version": "1.2",
        "cohort_id": "D1-TEST001",
        "cohort_manifest_ref": "D1-TEST001@d1-v1",
        "protected_period_architecture_hash": "2" * 64,
        "data_version": "d1-v1",
        "freeze_ts_utc": "2026-09-29T00:00:00Z",
        "hypothesis_versions": ["HYP-TEST001@1"],
        "family_ids": ["FAM-TEST001"],
        "evidence_budget_ids": ["EB-TEST001"],
        "outcome_access_roster": ["CONFIRMATION_RUNNER:runner-001"],
        "code_commit": "abc1234",
        "artifact_hashes": {"registration": "0" * 64},
        "unblinding_rule": "one-use registered run after frozen feasibility gate",
        "status": "FROZEN",
    }
    row.update(overrides)
    return with_freeze_hash(row)



def _cohort(**overrides):
    row = {
        "cohort_id": "D1-TEST001",
        "role": "CONFIRMATION",
        "status": "FROZEN",
        "start_date_utc": "2026-07-19",
        "end_date_utc": "2026-08-31",
        "data_version": "d1-v1",
        "canonical_builder_commit": "abc1234",
        "schema_version": "market-events-v1",
        "source_manifest_hashes": {"file_index": "3" * 64},
        "timezone_contract": "UTC",
        "unit_contract": "epoch nanoseconds int64",
        "known_coverage_gaps": [],
        "intended_sequence": "first protected confirmation cohort",
        "quarantine_status": "OUTCOMES_SEALED",
        "freeze_ts_utc": "2026-09-29T00:00:00Z",
    }
    row.update(overrides)
    return with_freeze_hash(row)

def test_freeze_hash_detects_mutation():
    row = _hyp()
    verify_freeze_hash(row)
    row["scientific_proposition"] = "mutated"
    with pytest.raises(ReproducibilityError, match="freeze_hash mismatch"):
        verify_freeze_hash(row)


def test_substantive_idempotence_can_ignore_only_explicit_volatile_field():
    assert_substantive_equal(
        {"rows": 7, "generated_at_utc": "a"},
        {"rows": 7, "generated_at_utc": "b"},
        volatile_fields={"generated_at_utc"},
    )
    with pytest.raises(ReproducibilityError):
        assert_substantive_equal({"rows": 7}, {"rows": 8})


def test_append_only_prefix_detects_rewrite_or_deletion():
    old = b'{"a":1}\n'
    assert_append_only_prefix(old, old + b'{"b":2}\n')
    with pytest.raises(ReproducibilityError):
        assert_append_only_prefix(old, b'{"a":9}\n')


def test_jsonl_append_rejects_duplicate_version_identity(tmp_path: Path):
    p = tmp_path / "registry.jsonl"
    row = _hyp()
    append_jsonl(p, row, unique_key=lambda r: (r["hypothesis_id"], r["version"]))
    with pytest.raises(RegistryValidationError, match="duplicate registry key"):
        append_jsonl(p, row, unique_key=lambda r: (r["hypothesis_id"], r["version"]))
    assert len(read_jsonl(p)) == 1


def test_hypothesis_records_dataset_influence_field_but_theory_only_idea_may_have_none():
    validate_hypothesis_version(_hyp(datasets_that_influenced_design=[]))
    raw = dict(_hyp()); raw.pop("freeze_hash"); raw.pop("datasets_that_influenced_design")
    with pytest.raises(RegistryValidationError, match="datasets_that_influenced_design"):
        validate_hypothesis_version(with_freeze_hash(raw))


def test_same_version_material_mutation_is_forbidden():
    old = _hyp()
    new = _hyp(scientific_proposition="different proposition")
    with pytest.raises(RegistryValidationError, match="existing hypothesis version"):
        validate_material_change(old, new)


def test_new_hypothesis_version_may_change_material_spec_but_cannot_move_backwards():
    old = _hyp(version=1)
    new = _hyp(version=2, scientific_proposition="refined proposition")
    validate_material_change(old, new)
    with pytest.raises(RegistryValidationError, match="backwards"):
        validate_material_change(new, old)


def test_genealogy_rejects_unknown_parent():
    child = _hyp(hypothesis_id="HYP-CHILD01", parent_hypothesis_ids=["HYP-MISSING"])
    with pytest.raises(RegistryValidationError, match="unknown parent"):
        validate_genealogy([child])


def test_genealogy_rejects_cycle():
    a = _hyp(hypothesis_id="HYP-AAAA", parent_hypothesis_ids=["HYP-BBBB"])
    b = _hyp(hypothesis_id="HYP-BBBB", parent_hypothesis_ids=["HYP-AAAA"])
    with pytest.raises(RegistryValidationError, match="cycle"):
        validate_genealogy([a, b])


def test_registered_candidate_requires_complete_position_economic_and_inference_spec():
    with pytest.raises(RegistryValidationError, match="incomplete"):
        validate_registered_spec(_hyp())
    validate_registered_spec(_registered_hyp())


def test_registered_candidate_requires_fresh_context_adversarial_review():
    row = _registered_hyp()
    row = dict(row)
    row.pop("freeze_hash")
    row["adversarial_review"] = dict(row["adversarial_review"], participated_in_discovery=True)
    row = with_freeze_hash(row)
    with pytest.raises(RegistryValidationError, match="discovery/authoring"):
        validate_registered_spec(row)


def test_registered_candidate_requires_mandatory_risk_gate():
    row = _registered_hyp()
    raw = dict(row)
    raw.pop("freeze_hash")
    raw["material_spec"] = dict(raw["material_spec"], risk_gates=[])
    row = with_freeze_hash(raw)
    with pytest.raises(RegistryValidationError, match="risk_gates"):
        validate_registered_spec(row)


def test_evidence_budget_requires_real_operating_guarantee():
    validate_evidence_budget(_budget())
    with pytest.raises(RegistryValidationError, match="control_method"):
        validate_evidence_budget(_budget(control_method="NONE"))


def test_single_primary_nominal_rule_cannot_cover_multiple_primary_claims():
    bad = _budget(primary_claim_ids=["HYP-TEST001@1", "HYP-TEST002@1"])
    with pytest.raises(RegistryValidationError, match="one primary"):
        validate_evidence_budget(bad)


def test_evidence_budget_requires_anti_family_splitting_rationale_and_do_not_spend_rule():
    with pytest.raises(RegistryValidationError, match="anti_family"):
        validate_evidence_budget(_budget(anti_family_splitting_rationale=""))
    with pytest.raises(RegistryValidationError, match="do_not_spend"):
        validate_evidence_budget(_budget(do_not_spend_threshold=""))


def test_confirmation_batch_must_be_finite_frozen_and_hashed():
    validate_confirmation_batch(_batch())
    with pytest.raises(RegistryValidationError, match="FROZEN"):
        validate_confirmation_batch(_batch(status="DRAFT"))


def test_confirmation_batch_rejects_duplicate_hypothesis_versions():
    with pytest.raises(RegistryValidationError, match="duplicate"):
        validate_confirmation_batch(_batch(hypothesis_versions=["HYP-TEST001@1", "HYP-TEST001@1"]))


def test_lifecycle_requires_ordered_promotion():
    validate_event_history([
        {"event_id": "HE-TEST001", "event_ts_utc": "2026-09-29T00:00:00Z", "rationale": "created", "hypothesis_version": "HYP-TEST001@1", "event_type": "PROMOTE", "to_stage": "IDEA"},
        {"event_id": "HE-TEST002", "event_ts_utc": "2026-09-29T00:01:00Z", "rationale": "feasible", "hypothesis_version": "HYP-TEST001@1", "event_type": "PROMOTE", "to_stage": "FEASIBLE"},
    ])
    with pytest.raises(RegistryValidationError, match="exactly one"):
        validate_event_history([
            {"event_id": "HE-TEST003", "event_ts_utc": "2026-09-29T00:00:00Z", "rationale": "created", "hypothesis_version": "HYP-TEST001@1", "event_type": "PROMOTE", "to_stage": "IDEA"},
            {"event_id": "HE-TEST004", "event_ts_utc": "2026-09-29T00:01:00Z", "rationale": "invalid skip", "hypothesis_version": "HYP-TEST001@1", "event_type": "PROMOTE", "to_stage": "REGISTERED_CANDIDATE"},
        ])


def test_terminal_classification_requires_existing_lifecycle_stage():
    with pytest.raises(RegistryValidationError, match="establish IDEA"):
        validate_event_history([{
            "event_id": "HE-NOIDEA1", "event_ts_utc": "2026-09-29T00:00:00Z",
            "rationale": "cannot classify invisible version", "hypothesis_version": "HYP-TEST001@1",
            "event_type": "CLASSIFY", "classification": "FAIL_EVIDENTIALLY_UNSUPPORTED",
        }])


def test_terminal_version_cannot_be_rescued_in_place():
    with pytest.raises(RegistryValidationError, match="new version"):
        validate_event_history([
            {"event_id": "HE-TEST005", "event_ts_utc": "2026-09-29T00:00:00Z", "rationale": "created", "hypothesis_version": "HYP-TEST001@1", "event_type": "PROMOTE", "to_stage": "IDEA"},
            {"event_id": "HE-TEST006", "event_ts_utc": "2026-09-29T00:01:00Z", "rationale": "failed", "hypothesis_version": "HYP-TEST001@1", "event_type": "CLASSIFY", "classification": "FAIL_EVIDENTIALLY_UNSUPPORTED"},
            {"event_id": "HE-TEST007", "event_ts_utc": "2026-09-29T00:02:00Z", "rationale": "improper rescue", "hypothesis_version": "HYP-TEST001@1", "event_type": "PROMOTE", "to_stage": "FEASIBLE"},
        ])


def test_discovery_agent_cannot_access_d1_even_for_integrity_metadata():
    got = authorize_protected_access(role="DISCOVERY_AGENT", zone="D1", content_class="INTEGRITY_METADATA")
    assert not got.allowed


def test_feasibility_causal_access_requires_registered_gate():
    denied = authorize_protected_access(
        role="FEASIBILITY_AGENT", zone="D1", content_class="CAUSAL_FIELDS_FOR_REGISTERED_GATE", registered_gate=False
    )
    allowed = authorize_protected_access(
        role="FEASIBILITY_AGENT", zone="D1", content_class="CAUSAL_FIELDS_FOR_REGISTERED_GATE", registered_gate=True
    )
    assert not denied.allowed and allowed.allowed


def test_detailed_d1_feasibility_marks_role_design_contaminated():
    got = authorize_protected_access(
        role="FEASIBILITY_AGENT", zone="D1", content_class="DETAILED_FEASIBILITY_DIAGNOSTICS", registered_gate=True
    )
    assert got.allowed and got.contamination_after == "DESIGN_CONTAMINATED"
    with pytest.raises(RegistryValidationError, match="contaminated"):
        assert_blind_design_eligible(got.contamination_after)


def test_contamination_is_monotone_and_cannot_be_downgraded():
    assert merge_contamination("OUTCOME_CONTAMINATED", "CLEAN") == "OUTCOME_CONTAMINATED"
    assert merge_contamination("CLEAN", "DESIGN_CONTAMINATED") == "DESIGN_CONTAMINATED"


def test_p1_outcomes_are_denied_even_to_confirmation_runner_before_d1_freeze():
    got = authorize_protected_access(
        role="CONFIRMATION_RUNNER",
        zone="P1_PROTECTED",
        content_class="FUTURE_OUTCOMES",
        batch_frozen=True,
        outcome_access_authorized=True,
        clean_confirmation_context=True,
    )
    assert not got.allowed


def test_d1_outcomes_require_frozen_authorized_clean_confirmation_context():
    denied = authorize_protected_access(
        role="CONFIRMATION_RUNNER", zone="D1", content_class="FUTURE_OUTCOMES", batch_frozen=False
    )
    allowed = authorize_protected_access(
        role="CONFIRMATION_RUNNER",
        zone="D1",
        content_class="FUTURE_OUTCOMES",
        batch_frozen=True,
        outcome_access_authorized=True,
        clean_confirmation_context=True,
    )
    assert not denied.allowed
    assert allowed.allowed and allowed.contamination_after == "OUTCOME_CONTAMINATED"


def test_predictor_guard_rejects_governance_and_future_columns():
    with pytest.raises(RegistryValidationError, match="governance"):
        assert_predictor_columns_safe(["trade_count", "overlap_component_15m"])
    with pytest.raises(RegistryValidationError, match="future"):
        assert_predictor_columns_safe(["trade_count", "p75_reached"])


def test_eligibility_guard_rejects_future_completeness_or_outcome_fields():
    assert_eligibility_fields_safe(["clock_available", "spread_bps"])
    with pytest.raises(RegistryValidationError, match="future"):
        assert_eligibility_fields_safe(["future_outcome_available"])


def test_random_cross_validation_is_blocked_and_purged_walk_forward_allowed():
    with pytest.raises(RegistryValidationError, match="random"):
        assert_time_respecting_validation("random_split")
    assert_time_respecting_validation("purged_walk_forward")



def test_protected_cohort_requires_immutable_architecture_fields_and_hashes():
    validate_protected_cohort(_cohort())
    with pytest.raises(RegistryValidationError, match="source_manifest_hashes"):
        validate_protected_cohort(_cohort(source_manifest_hashes={}))


def test_access_event_ledger_rejects_contamination_downgrade():
    events = [
        {
            "access_event_id": "PA-TEST001", "event_ts_utc": "2026-09-29T00:00:00Z",
            "actor_id": "reviewer-1", "role": "FEASIBILITY_AGENT", "cohort_id": "D1-TEST001",
            "content_class": "DETAILED_FEASIBILITY_DIAGNOSTICS", "purpose": "registered gate",
            "allowed": True, "contamination_before": "CLEAN", "contamination_after": "DESIGN_CONTAMINATED",
        },
        {
            "access_event_id": "PA-TEST002", "event_ts_utc": "2026-09-29T00:01:00Z",
            "actor_id": "reviewer-1", "role": "FEASIBILITY_AGENT", "cohort_id": "D1-TEST001",
            "content_class": "INTEGRITY_METADATA", "purpose": "follow-up",
            "allowed": True, "contamination_before": "DESIGN_CONTAMINATED", "contamination_after": "CLEAN",
        },
    ]
    with pytest.raises(RegistryValidationError, match="downgraded"):
        validate_access_event_history(events)


def test_decision_log_requires_explicit_outcome_exposure_state():
    good = [{
        "decision_id": "DEC-TEST001", "decision_ts_utc": "2026-09-29T00:00:00Z",
        "decision": "keep P1 sealed", "rationale": "preserve independent evidence",
        "relevant_outcomes_seen": False, "datasets_considered": ["D0"],
    }]
    validate_decision_log(good)
    bad = [dict(good[0])]; bad[0].pop("relevant_outcomes_seen")
    with pytest.raises(RegistryValidationError, match="relevant_outcomes_seen"):
        validate_decision_log(bad)


def test_design_contaminated_confirmation_role_cannot_claim_clean_first_unblinding():
    got = authorize_protected_access(
        role="CONFIRMATION_RUNNER", zone="D1", content_class="FUTURE_OUTCOMES",
        current_contamination="DESIGN_CONTAMINATED", batch_frozen=True,
        outcome_access_authorized=True, clean_confirmation_context=True,
    )
    assert not got.allowed

def test_contract_closes_p1_and_outcome_firewall_and_forbids_family_splitting():
    c = json.loads((REPO_ROOT / "research" / "governance" / "research_registry_contract_v1.json").read_text())
    assert c["outcome_firewall"]["p1_read"] is False
    assert c["outcome_firewall"]["future_outcome_values_read"] is False
    assert c["evidence_control"]["anti_family_splitting_rationale_required"] is True
    assert c["protected_access"]["contamination_is_monotone"] is True


def test_initial_r07_registries_are_valid_and_create_no_hypotheses_or_batches():
    m = _validator()
    summary = m.validate_installation()
    assert summary["status"] == "PASS"
    assert summary["p1_status"] == "SEALED_OUTCOME_UNSEEN"
    assert summary["registry_counts"]["hypothesis_versions"] == 0
    assert summary["registry_counts"]["confirmation_batches"] == 0


def test_r07_self_test_passes_without_opening_market_or_outcome_data():
    m = _validator()
    m.self_test()


def test_manifest_declares_zero_hypotheses_and_zero_p1_access():
    m = json.loads((REPO_ROOT / "research" / "r0" / "registries" / "R0_7_IMPLEMENTATION_MANIFEST.json").read_text())
    assert m["hypotheses_created"] == 0
    assert m["confirmation_batches_created"] == 0
    assert m["protected_cohorts_created"] == 0
    assert m["p1_read"] is False
    assert m["future_outcomes_read"] is False
