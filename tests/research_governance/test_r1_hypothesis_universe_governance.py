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

from research_core.hypothesis_universe import (  # noqa: E402
    DEFAULT_GENERATION_LENSES,
    assert_generation_not_testing,
    assert_no_protected_influence,
    assert_testability_separate_from_merit,
    promote_idea_to_hypothesis,
    validate_family,
    validate_generation_session,
    validate_idea,
    validate_universe,
)
from research_core.registry import (  # noqa: E402
    RegistryValidationError,
    read_jsonl,
    validate_hypothesis_version,
    validate_registered_spec,
    with_freeze_hash,
)


def _validator():
    p = REPO_ROOT / "research" / "r1" / "universe" / "validate_hypothesis_universe_governance.py"
    spec = importlib.util.spec_from_file_location("r10_validator", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _family(**overrides):
    row = {
        "family_id": "FAM-TEST001",
        "family_name": "Test mechanism",
        "created_at_utc": "2026-10-09T00:00:00Z",
        "mechanism_class": "FLOW_IMBALANCE",
        "mechanism_statement": "A causal flow state changes the future executable return distribution.",
        "market_state_domains": ["PERPS_TRADES"],
        "relationship_types": ["MARGINAL"],
        "variant_dimensions_kept_in_family": ["threshold", "horizon", "lookback_window"],
        "new_family_boundary_rule": "Create a new family only when the economic mechanism or causal relationship changes.",
        "anti_family_splitting_rationale": "Threshold and horizon variants are one mechanism, not independent claims.",
        "legacy_lineage_refs": [],
    }
    row.update(overrides)
    return with_freeze_hash(row)


def _idea(**overrides):
    row = {
        "idea_id": "IDEA-R1-TEST001",
        "version": 1,
        "family_id": "FAM-TEST001",
        "created_at_utc": "2026-10-09T00:01:00Z",
        "title": "Test causal-flow idea",
        "scientific_proposition": "A causally observed flow state predicts an executable future return distribution.",
        "economic_mechanism": "Persistent demand may continue to consume available opposing liquidity.",
        "relationship_type": "MARGINAL",
        "expected_market_response": "Directional continuation conditional on the causal state.",
        "causal_state_requirements": [
            {"state_family": "PERPS_TRADES", "role": "aggressive-flow state", "required": True}
        ],
        "trigger_concept": "A pre-decision flow state crosses a conceptual trigger.",
        "candidate_definition_status": "TO_BE_DEVELOPED_IN_D0",
        "position_concept": {
            "directionality": "state-aligned",
            "entry_style": "marketable after governed decision latency",
            "horizon_concept": "fixed short-horizon development exit",
            "exit_concept": "time exit; stop/target deferred to D0 development",
        },
        "primary_economic_target": "Positive net expectancy after explicit trading frictions.",
        "risk_concepts": ["MAE", "tail loss", "episode concentration"],
        "execution_dependencies": ["marketable execution", "explicit latency", "explicit costs"],
        "generation_mode": "OUTCOME_BLIND",
        "source_refs": [
            {
                "source_type": "MARKET_MECHANISM",
                "source_zone": "NONE",
                "reference": "Synthetic test mechanism",
                "result_informed": False,
            }
        ],
        "parent_idea_refs": [],
        "parent_hypothesis_ids": [],
        "legacy_lineage_refs": [],
        "derived_from_results": [],
        "datasets_that_influenced_design": [],
        "infrastructure_assessment": {
            "r2_testability": "TESTABLE_NOW",
            "available_capabilities": ["PERPS_TRADES"],
            "missing_capabilities": [],
            "assessment_basis": "R0.5 provides governed PERPS_TRADES causal state.",
        },
    }
    row.update(overrides)
    return with_freeze_hash(row)


def _session(version: int = 1, **overrides):
    row = {
        "session_id": "GEN-R1-TEST001",
        "version": version,
        "created_at_utc": "2026-10-09T00:00:00Z",
        "status": "FROZEN_PLANNED",
        "planned_lenses": list(DEFAULT_GENERATION_LENSES),
        "completed_lenses": [],
        "allowed_source_zones": ["NONE", "PUBLIC", "D0", "LEGACY_D0"],
        "allowed_generation_modes": ["OUTCOME_BLIND", "D0_RESULT_INFORMED", "LEGACY_RESULT_INFORMED"],
        "p1_market_data_read": False,
        "p1_outcome_read": False,
        "protected_performance_read": False,
        "no_outcome_based_pruning_during_generation": True,
        "retain_failed_duplicate_and_legacy_lineage": True,
        "family_splitting_rule": "Threshold/horizon/window variants stay in-family unless the mechanism changes.",
        "stopping_rule": {
            "type": "PREDECLARED_LENS_COVERAGE",
            "required_lenses": list(DEFAULT_GENERATION_LENSES),
            "completion_criterion": "All lenses completed or explicitly waived before R1.9.",
        },
    }
    row.update(overrides)
    return with_freeze_hash(row)


def test_family_requires_anti_splitting_rule_and_variant_dimensions():
    validate_family(_family())
    with pytest.raises(RegistryValidationError, match="variant_dimensions"):
        validate_family(_family(variant_dimensions_kept_in_family=[]))
    with pytest.raises(RegistryValidationError, match="anti_family"):
        validate_family(_family(anti_family_splitting_rationale=""))


def test_family_rejects_ungoverned_extra_fields():
    with pytest.raises(RegistryValidationError, match="ungoverned fields"):
        validate_family(_family(observed_alpha_bps=9.0))


def test_outcome_blind_idea_cannot_hide_result_influence():
    bad_source = [{"source_type": "D0_RESULT", "source_zone": "D0", "reference": "result-x", "result_informed": True}]
    with pytest.raises(RegistryValidationError, match="OUTCOME_BLIND"):
        validate_idea(_idea(source_refs=bad_source, derived_from_results=["result-x"]))


def test_d0_result_informed_idea_requires_result_and_dataset_lineage():
    src = [{"source_type": "D0_RESULT", "source_zone": "D0", "reference": "result-x", "result_informed": True}]
    good = _idea(
        generation_mode="D0_RESULT_INFORMED",
        source_refs=src,
        derived_from_results=["D0:result-x"],
        datasets_that_influenced_design=["D0:market_events_v1"],
    )
    validate_idea(good)
    with pytest.raises(RegistryValidationError, match="derived_from_results"):
        validate_idea(_idea(generation_mode="D0_RESULT_INFORMED", source_refs=src))


def test_legacy_result_informed_idea_preserves_legacy_source():
    row = _idea(
        generation_mode="LEGACY_RESULT_INFORMED",
        source_refs=[{
            "source_type": "LEGACY_LINEAGE",
            "source_zone": "LEGACY_D0",
            "reference": "7M.8 funding counter-crowding result",
            "result_informed": True,
        }],
        legacy_lineage_refs=["LEGACY-7M.8"],
        derived_from_results=["LEGACY-7M.8:funding"],
        datasets_that_influenced_design=["LEGACY-D0"],
    )
    validate_idea(row)


def test_protected_source_zone_is_rejected():
    raw = dict(_idea()); raw.pop("freeze_hash")
    raw["source_refs"] = [{
        "source_type": "MARKET_MECHANISM", "source_zone": "P1_PROTECTED",
        "reference": "forbidden", "result_informed": False,
    }]
    with pytest.raises(RegistryValidationError, match="protected P1/D1"):
        validate_idea(with_freeze_hash(raw))


def test_guard_rejects_protected_influence_even_outside_schema_validation():
    with pytest.raises(RegistryValidationError, match="protected P1/D1"):
        assert_no_protected_influence({"source_refs": [{"source_zone": "D1"}]})


def test_idea_packet_cannot_embed_fresh_outcome_statistics():
    with pytest.raises(RegistryValidationError, match="ungoverned fields"):
        validate_idea(_idea(p_value=0.01))
    with pytest.raises(RegistryValidationError, match="outcome-test"):
        assert_generation_not_testing({"observed_return": 4.0})


def test_testability_is_separate_from_scientific_merit():
    assert_testability_separate_from_merit(_idea())
    with pytest.raises(RegistryValidationError, match="not an infrastructure field"):
        assert_testability_separate_from_merit({
            "infrastructure_assessment": {"scientific_merit_score": 0.9}
        })


def test_blocked_idea_must_name_missing_capability():
    raw = dict(_idea()); raw.pop("freeze_hash")
    raw["infrastructure_assessment"] = {
        "r2_testability": "BLOCKED_PENDING_GOVERNED_STATE",
        "available_capabilities": [],
        "missing_capabilities": [],
        "assessment_basis": "requires order book state",
    }
    with pytest.raises(RegistryValidationError, match="missing governed capability"):
        validate_idea(with_freeze_hash(raw))


def test_testable_now_cannot_claim_missing_capability():
    raw = dict(_idea()); raw.pop("freeze_hash")
    raw["infrastructure_assessment"] = {
        "r2_testability": "TESTABLE_NOW",
        "available_capabilities": ["PERPS_TRADES"],
        "missing_capabilities": ["ORDER_BOOK"],
        "assessment_basis": "contradictory fixture",
    }
    with pytest.raises(RegistryValidationError, match="TESTABLE_NOW"):
        validate_idea(with_freeze_hash(raw))


def test_universe_rejects_unknown_family_reference():
    with pytest.raises(RegistryValidationError, match="unknown family"):
        validate_universe([], [_idea()], [_session()])


def test_idea_version_must_preserve_prior_version_parent():
    v1 = _idea()
    v2 = _idea(version=2, title="Revised idea")
    with pytest.raises(RegistryValidationError, match="prior version"):
        validate_universe([_family()], [v1, v2], [_session()])
    raw = dict(v2); raw.pop("freeze_hash"); raw["parent_idea_refs"] = ["IDEA-R1-TEST001@1"]
    validate_universe([_family()], [v1, with_freeze_hash(raw)], [_session()])


def test_idea_genealogy_rejects_cycle():
    a = _idea(idea_id="IDEA-R1-AAAA", parent_idea_refs=["IDEA-R1-BBBB@1"])
    b = _idea(idea_id="IDEA-R1-BBBB", parent_idea_refs=["IDEA-R1-AAAA@1"])
    with pytest.raises(RegistryValidationError, match="cycle"):
        validate_universe([_family()], [a, b], [_session()])


def test_generation_plan_freezes_all_predeclared_lenses():
    validate_generation_session(_session())
    assert len(DEFAULT_GENERATION_LENSES) == 9
    bad = list(DEFAULT_GENERATION_LENSES[:-1])
    with pytest.raises(RegistryValidationError, match="frozen default lens order"):
        validate_generation_session(_session(planned_lenses=bad, stopping_rule={
            "type": "PREDECLARED_LENS_COVERAGE",
            "required_lenses": bad,
            "completion_criterion": "bad fixture",
        }))


def test_complete_session_requires_every_planned_lens():
    with pytest.raises(RegistryValidationError, match="every planned lens"):
        validate_generation_session(_session(status="COMPLETE", completed_lenses=[DEFAULT_GENERATION_LENSES[0]]))
    validate_generation_session(_session(status="COMPLETE", completed_lenses=list(DEFAULT_GENERATION_LENSES)))


def test_generation_session_cannot_enable_outcome_pruning():
    with pytest.raises(RegistryValidationError, match="prune"):
        validate_generation_session(_session(no_outcome_based_pruning_during_generation=False))


def test_session_versions_cannot_change_frozen_lenses_or_remove_completion():
    s1 = _session()
    s2 = _session(version=2, status="IN_PROGRESS", completed_lenses=[DEFAULT_GENERATION_LENSES[0]])
    validate_universe([], [], [s1, s2])
    changed = _session(version=2, status="IN_PROGRESS", planned_lenses=list(reversed(DEFAULT_GENERATION_LENSES)), completed_lenses=[] , stopping_rule={
        "type": "PREDECLARED_LENS_COVERAGE",
        "required_lenses": list(DEFAULT_GENERATION_LENSES),
        "completion_criterion": "bad fixture",
    })
    with pytest.raises(RegistryValidationError):
        validate_universe([], [], [s1, changed])
    s3 = _session(version=3, status="IN_PROGRESS", completed_lenses=[])
    with pytest.raises(RegistryValidationError, match="cannot be removed"):
        validate_universe([], [], [s1, s2, s3])


def test_promotion_preserves_lineage_and_produces_r07_compatible_idea_packet():
    idea = _idea(
        parent_hypothesis_ids=["HYP-OLD001"],
        legacy_lineage_refs=["LEGACY-7M"],
        datasets_that_influenced_design=["D0"],
    )
    promoted = promote_idea_to_hypothesis(
        idea,
        hypothesis_id="HYP-R1TEST001",
        created_at_utc="2026-10-09T00:05:00Z",
    )
    validate_hypothesis_version(promoted)
    assert promoted["parent_hypothesis_ids"] == ["HYP-OLD001"]
    assert promoted["datasets_that_influenced_design"] == ["D0"]
    assert promoted["material_spec"]["legacy_lineage_refs"] == ["LEGACY-7M"]
    assert promoted["material_spec"]["r1_idea_ref"] == "IDEA-R1-TEST001@1"


def test_promotion_does_not_bypass_registered_candidate_gate():
    promoted = promote_idea_to_hypothesis(
        _idea(), hypothesis_id="HYP-R1TEST002", created_at_utc="2026-10-09T00:05:00Z"
    )
    with pytest.raises(RegistryValidationError, match="incomplete"):
        validate_registered_spec(promoted)


def test_contract_anchors_to_frozen_r0_and_keeps_p1_closed():
    obj = json.loads((REPO_ROOT / "research/governance/hypothesis_universe_contract_v1.json").read_text())
    assert obj["accepted_r0_parent"]["commit"] == "71192f37e8aa44da4c30326048debf65b1ca5792"
    assert obj["accepted_r0_parent"]["tag"] == "r0-foundation-accepted-v1"
    assert obj["accepted_r0_parent"]["r0_9_substantive_audit_hash"] == "3faab2f04bb4902d19d90a0a9c6df49b42e62b648572c6316110ea869b87accd"
    assert obj["protected_data_firewall"]["p1_market_data_read"] is False
    assert obj["protected_data_firewall"]["p1_outcome_read"] is False


def test_contract_forbids_ad_hoc_testing_of_unsupported_state_families():
    obj = json.loads((REPO_ROOT / "research/governance/hypothesis_universe_contract_v1.json").read_text())
    assert obj["idea_policy"]["unsupported_market_state_families_may_be_proposed"] is True
    assert obj["idea_policy"]["unsupported_market_state_families_may_be_tested_ad_hoc_in_r2"] is False


def test_repository_generation_plan_is_valid_and_first_record_is_frozen():
    rows = read_jsonl(REPO_ROOT / "research/governance/universe/generation_sessions.jsonl")
    assert rows
    validate_generation_session(rows[0])
    assert rows[0]["session_id"] == "GEN-R1-UNIVERSE-V1"
    assert rows[0]["status"] == "FROZEN_PLANNED"
    assert rows[0]["planned_lenses"] == list(DEFAULT_GENERATION_LENSES)


def test_r10_manifest_declares_no_testing_no_outcomes_and_no_registry_mutation():
    obj = json.loads((REPO_ROOT / "research/r1/universe/R1_0_IMPLEMENTATION_MANIFEST.json").read_text())
    assert obj["creates_market_hypotheses"] is False
    assert obj["tests_market_hypotheses"] is False
    assert obj["reads_d0_outcomes"] is False
    assert obj["p1_market_data_read"] is False
    assert obj["p1_outcome_read"] is False
    assert obj["mutates_r0_7_hypothesis_registry"] is False


def test_validator_source_does_not_reference_market_or_outcome_paths():
    text = (REPO_ROOT / "research/r1/universe/validate_hypothesis_universe_governance.py").read_text().lower()
    assert "data/canonical" not in text
    assert "outcomes.parquet" not in text
    assert "future/outcomes" not in text


def test_living_research_readme_records_r10_and_frozen_r0_checkpoint():
    text = (REPO_ROOT / "research/README.md").read_text(encoding="utf-8")
    assert "R1.0" in text
    assert "Hypothesis-generation governance" in text
    assert "71192f37e8aa44da4c30326048debf65b1ca5792" in text
    assert "r0-foundation-accepted-v1" in text


def test_r1_readme_predeclares_sequence_through_r19():
    text = (REPO_ROOT / "research/r1/README.md").read_text(encoding="utf-8")
    for step in ("R1.0", "R1.1", "R1.2", "R1.3", "R1.4", "R1.5", "R1.6", "R1.7", "R1.8", "R1.9"):
        assert step in text


def test_validator_installation_passes_without_market_files():
    summary = _validator().validate_installation()
    assert summary["status"] == "PASS"
    assert summary["p1_market_data_read"] is False
    assert summary["p1_outcome_read"] is False
    assert summary["d0_outcomes_read_by_r1_0"] is False
    assert summary["predeclared_lenses"] == list(DEFAULT_GENERATION_LENSES)


def test_validator_self_test_passes():
    _validator().self_test()
